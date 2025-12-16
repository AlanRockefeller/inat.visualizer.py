#!/home/alan/anaconda3/envs/inat_env/bin/python
# Run `conda activate inat_env` before executing this script to ensure the correct environment is used.

# Requires:  libxcb-cursor0 libxkbcommon-x11-0

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # run Qt via XWayland (usually avoids the Wayland protocol crash)


import sys
import argparse
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QLineEdit, QPushButton, QComboBox, QMenuBar,
                             QStatusBar, QFileDialog, QLabel, QListWidget, QMessageBox,
                             QProgressBar, QColorDialog, QTextEdit, QSplashScreen, QDialog,
                             QDialogButtonBox, QSlider, QSpinBox, QSizePolicy)
from PyQt6.QtCore import QSettings, Qt, QTimer, QThread, pyqtSignal, QObject, QMutex, QWaitCondition, QMutexLocker
from PyQt6.QtGui import QAction, QFont, QColor, QPixmap, QDoubleValidator, QGuiApplication
import pyinaturalist
import pandas as pd
from datetime import datetime
import os
import time
import requests
from requests.exceptions import HTTPError
from urllib.parse import urlencode
import duckdb
import math
import importlib
import platform
import json
import logging
from pathlib import Path

# Set matplotlib backend to PyQt6 before importing matplotlib
import matplotlib
matplotlib.use('QtAgg')  # Explicitly use PyQt6 backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import requests
from PIL import Image
from io import BytesIO

from requests.exceptions import RequestException
from collections import OrderedDict

# Configure pyinaturalist with User-Agent
pyinaturalist.user_agent = "iNaturalist Seasonal Visualizer/1.0 (Contact: your_email@example.com)"

# --- Constants for MapDialog ---
MAX_TILES_PER_REQUEST = 100
MAX_CACHE_SIZE = 500  # Max number of tiles to keep in memory (RAM)
TILE_CACHE_DIR_NAME = "tile_cache"
MAX_DISK_CACHE_SIZE_MB = 200

class DiskTileCache:
    """Persistent disk cache for map tiles with size limit and LRU pruning."""
    def __init__(self, parent_dir, max_size_mb=MAX_DISK_CACHE_SIZE_MB):
        self.cache_dir = Path(parent_dir) / TILE_CACHE_DIR_NAME
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, z, x, y):
        return self.cache_dir / str(z) / str(x) / f"{y}.png"

    def get_tile(self, z, x, y):
        path = self.get_path(z, x, y)
        if path.exists():
            try:
                # Update mtime for LRU
                path.touch()
                return Image.open(path).convert("RGB")
            except Exception as e:
                logging.warning(f"Corrupt tile in disk cache {path}: {e}")
        return None

    def put_tile(self, z, x, y, image_bytes):
        path = self.get_path(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        temp_path = path.with_suffix(f".tmp.{os.getpid()}.{time.time()}")
        try:
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            os.replace(temp_path, path)
            self._prune_if_needed()
        except Exception as e:
            logging.error(f"Failed to write tile to disk {path}: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _prune_if_needed(self):
        # Simple probability-based check to avoid scanning on every write
        if np.random.random() > 0.05: 
            return

        total_size = 0
        files = []
        for p in self.cache_dir.rglob("*.png"):
            try:
                stat = p.stat()
                total_size += stat.st_size
                files.append((stat.st_mtime, p, stat.st_size))
            except OSError:
                pass

        if total_size > self.max_size_mb * 1024 * 1024:
            # Sort by mtime (oldest first)
            files.sort()
            deleted_size = 0
            target_reduction = total_size * 0.2 # clear 20%
            for mtime, p, size in files:
                try:
                    p.unlink()
                    deleted_size += size
                    if deleted_size >= target_reduction:
                        break
                except OSError:
                    pass
            # Try to remove empty dirs
            for p in self.cache_dir.rglob("*"):
                if p.is_dir():
                    try:
                        p.rmdir() # Only removes if empty
                    except OSError:
                        pass

class TileLoaderWorker(QThread):
    """Background worker to fetch map tiles."""
    view_ready = pyqtSignal(int, object, tuple)  # job_id, composite_image, extent

    def __init__(self, working_dir):
        super().__init__()
        self.disk_cache = DiskTileCache(working_dir)
        self.ram_cache = OrderedDict()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._pending_job = None  # (job_id, zoom, x_min, x_max, y_min, y_max)
        self._running = True
        
    def request_view(self, job_id, zoom, x_min, x_max, y_min, y_max):
        with QMutexLocker(self.mutex):
            self._pending_job = (job_id, zoom, x_min, x_max, y_min, y_max)
            self.wait_condition.wakeOne()

    def stop(self):
        with QMutexLocker(self.mutex):
            self._running = False
            self.wait_condition.wakeOne()
        self.wait()

    def run(self):
        session = requests.Session()
        session.headers.update({"User-Agent": pyinaturalist.user_agent})

        while True:
            job = None
            with QMutexLocker(self.mutex):
                while self._pending_job is None and self._running:
                    self.wait_condition.wait(self.mutex)
                if not self._running:
                    return
                job = self._pending_job
                self._pending_job = None # Clear it so we don't repeat unless new one comes
            
            if not job: 
                continue

            job_id, zoom, x_min, x_max, y_min, y_max = job
            self._process_job(session, job_id, zoom, x_min, x_max, y_min, y_max)

    def _process_job(self, session, job_id, zoom, x_min, x_max, y_min, y_max):
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)
        
        tiles = []
        tile_positions = []
        
        for x in x_range:
            for y in y_range:
                # Basic stale check
                with QMutexLocker(self.mutex):
                    if self._pending_job is not None and self._pending_job[0] > job_id:
                        return # Abort this job

                tile_key = (zoom, x, y)
                img = None
                
                # 1. RAM Cache
                if tile_key in self.ram_cache:
                    img = self.ram_cache[tile_key]
                    self.ram_cache.move_to_end(tile_key)
                
                # 2. Disk Cache
                if not img:
                    img = self.disk_cache.get_tile(zoom, x, y)
                    if img:
                        self.ram_cache[tile_key] = img
                        if len(self.ram_cache) > MAX_CACHE_SIZE:
                            self.ram_cache.popitem(last=False)
                
                # 3. Network
                if not img:
                    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                    try:
                        resp = session.get(url, timeout=2.0)
                        if resp.status_code == 200:
                            img_bytes = resp.content
                            img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            
                            # Update caches
                            self.ram_cache[tile_key] = img
                            if len(self.ram_cache) > MAX_CACHE_SIZE:
                                self.ram_cache.popitem(last=False)
                            self.disk_cache.put_tile(zoom, x, y, img_bytes)
                        else:
                            logging.debug(f"Tile {zoom}/{x}/{y} HTTP {resp.status_code}")
                    except RequestException as e:
                        logging.debug(f"Failed to load tile {zoom}/{x}/{y}: {e}")

                if img:
                    tiles.append(img)
                    tile_positions.append((x, y))

        if not tiles:
            return

        # Stitch
        min_x = min(p[0] for p in tile_positions)
        min_y = min(p[1] for p in tile_positions)
        
        tile_w, tile_h = tiles[0].size
        w = (max(p[0] for p in tile_positions) - min_x + 1) * tile_w
        h = (max(p[1] for p in tile_positions) - min_y + 1) * tile_h
        
        composite = Image.new('RGB', (w, h), (230, 230, 230))
        for img, (x, y) in zip(tiles, tile_positions):
            composite.paste(img, ((x - min_x) * tile_w, (y - min_y) * tile_h))
            
        # Calculate extent
        def num2deg(xtile, ytile, zoom):
            n = 2.0 ** zoom
            lon_deg = xtile / n * 360.0 - 180.0
            lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
            lat_deg = np.degrees(lat_rad)
            return lat_deg, lon_deg

        north, west = num2deg(min_x, min_y, zoom)
        south, east = num2deg(max(p[0] for p in tile_positions) + 1, max(p[1] for p in tile_positions) + 1, zoom)
        
        # Emit result
        self.view_ready.emit(job_id, composite, (west, east, south, north))

class DatabaseProgressTracker:
    """Track database operation progress without impacting performance"""
    
    def __init__(self, progress_widget):
        self.progress_widget = progress_widget
        self.operation_start = None
        self.last_update = None
        self.estimated_total = None
        
    def start_operation(self, operation_name, estimated_total=None):
        """Start tracking a database operation"""
        self.operation_start = time.time()
        self.last_update = self.operation_start
        self.estimated_total = estimated_total
        self.progress_widget.start_progress(estimated_total or 0, f"Starting {operation_name}...")
        
    def update_progress(self, current_count=None, message=None):
        """Update progress during database operation"""
        current_time = time.time()
        
        if current_count is not None and self.estimated_total:
            progress = min(100, int((current_count / self.estimated_total) * 100))
            self.progress_widget.update_progress(progress, message)
        else:
            self.progress_widget.update_progress(message=message)
            
        self.last_update = current_time
        
    def finish_operation(self, final_count=None, message=None):
        """Finish tracking the operation"""
        if final_count is not None:
            self.progress_widget.finish_progress(f"{message or 'Operation completed'}: {final_count} results")
        else:
            self.progress_widget.finish_progress(message or "Operation completed")

class CustomSplashScreen(QSplashScreen):
    """Splash screen that scales to fit the screen without overflowing."""

    def __init__(self, image_path, parent=None, scale_factor=1.4):
        # Load the base image
        original_pixmap = QPixmap(image_path)

        # Get the screen geometry
        screen = QApplication.primaryScreen().geometry()
        screen_w = screen.width()
        screen_h = screen.height()

        # Compute target size based on desired scale factor
        target_w = original_pixmap.width() * scale_factor
        target_h = original_pixmap.height() * scale_factor

        # If too large, shrink so it never exceeds screen boundaries
        if target_w > screen_w or target_h > screen_h:
            scale_ratio = min(screen_w / original_pixmap.width(),
                              screen_h / original_pixmap.height())
            target_w = original_pixmap.width() * scale_ratio
            target_h = original_pixmap.height() * scale_ratio

        # Scale with smooth filtering
        scaled_pixmap = original_pixmap.scaled(
            int(target_w),
            int(target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        super().__init__(scaled_pixmap)

        # Window flags
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        # Center on screen
        self.move(
            screen.x() + (screen_w - self.width()) // 2,
            screen.y() + (screen_h - self.height()) // 2
        )

        # Status label
        self.status_label = QLabel(self)
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        self.status_label.setText("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Bottom-centered position
        self.status_label.resize(self.width() - 20, 50)
        self.status_label.move(10, self.height() - 70)

        self.show()
        QApplication.processEvents()

    def update_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()


class MapDialog(QDialog):
    """Interactive map dialog for setting coordinates and radius using matplotlib"""
    
    def __init__(self, parent=None, lat=37.7749, lon=-122.4194, radius=10):
        super().__init__(parent)
        self.parent = parent
        self.lat = lat
        self.lon = lon
        self.radius = radius
        
        # Worker setup
        working_dir = parent.working_dir if parent else os.getcwd()
        self.worker = TileLoaderWorker(working_dir)
        self.worker.view_ready.connect(self.on_view_ready)
        self.worker.start()
        
        self.job_counter = 0
        self.current_job_id = 0
        
        self.setWindowTitle("Interactive Map - Set Location")
        self.setModal(True)

        screen = QApplication.primaryScreen().geometry()
        w = int(screen.width() * 0.75)
        h = int(screen.height() * 0.75)
        self.resize(w, h)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        instructions = QLabel("Click on the map to set location. Use mouse wheel to zoom.")
        instructions.setStyleSheet("font-weight: bold; color: #333;")
        controls_layout.addWidget(instructions)
        
        radius_label = QLabel("Radius (km):")
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(1, 1000)
        self.radius_spinbox.setValue(int(radius))
        self.radius_spinbox.valueChanged.connect(self.update_radius)
        
        self.coord_label = QLabel(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        
        controls_layout.addWidget(radius_label)
        controls_layout.addWidget(self.radius_spinbox)
        controls_layout.addWidget(self.coord_label)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(800, 600)
        layout.addWidget(self.canvas, stretch=1)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Timer for debounced reloads
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200) # 200ms debounce
        self.update_timer.timeout.connect(self.request_tiles_for_current_view)
        
        # Initialize the map
        self.ax.set_xlim(self.lon - 5, self.lon + 5)
        self.ax.set_ylim(self.lat - 5, self.lat + 5)
        
        self.ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        self.ax.set_title('High-Resolution Interactive Map - Click to Set Location', fontsize=14, fontweight='bold')
        self.ax.grid(False)
        
        # Overlays
        self.marker, = self.ax.plot([], [], 'ro', markersize=12, markeredgecolor='darkred', markeredgewidth=3, zorder=10)
        self.circle, = self.ax.plot([], [], 'r-', linewidth=3, alpha=0.8, zorder=5)
        self.update_overlays()

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Pan support
        self._is_panning = False
        self._last_pan_pos = None
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Trigger initial load
        self.request_tiles_for_current_view()
        
    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)
        
    def request_tiles_for_current_view(self):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        
        # Calculate optimal zoom
        try:
            bbox = self.ax.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            width_px = bbox.width * self.figure.dpi
        except Exception:
            width_px = 800

        lon_span = abs(x1 - x0)
        if lon_span == 0 or width_px <= 0: return
        
        # Target: pixel density.
        target_scale = 1.5
        needed_2z = (target_scale * width_px * 360.0) / (256.0 * max(0.0001, lon_span))
        zoom = int(math.log2(needed_2z)) if needed_2z > 0 else 0
        zoom = max(0, min(zoom, 16))
        
        # Check tile budget
        def get_tile_range(z):
            n = 2.0 ** z
            xtile_min = int((x0 + 180.0) / 360.0 * n)
            xtile_max = int((x1 + 180.0) / 360.0 * n)
            
            def lat2y(lat):
                return (1.0 - math.asinh(math.tan(math.radians(max(-85, min(85, lat))))) / math.pi) / 2.0 * n
            
            y_a = int(lat2y(y1)) # y1 is max lat (top)
            y_b = int(lat2y(y0)) # y0 is min lat (bottom)
            return z, xtile_min, xtile_max, min(y_a, y_b), max(y_a, y_b)

        while zoom >= 0:
            z, x_min, x_max, y_min, y_max = get_tile_range(zoom)
            num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
            if num_tiles <= MAX_TILES_PER_REQUEST:
                break
            zoom -= 1
        
        # Submit job
        self.job_counter += 1
        self.current_job_id = self.job_counter
        self.worker.request_view(self.current_job_id, zoom, x_min, x_max, y_min, y_max)
        
    def on_view_ready(self, job_id, composite, extent):
        if job_id < self.current_job_id:
            return 
        
        for img in list(self.ax.images):
            img.remove()
        
        self.ax.imshow(np.array(composite), extent=extent, origin='upper', alpha=0.9)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Redraw overlays
        self.update_overlays()
        self.canvas.draw_idle()

    def update_overlays(self):
        self.marker.set_data([self.lon], [self.lat])
        
        circle_lons = []
        circle_lats = []
        for angle in np.linspace(0, 2*np.pi, 100):
            lat_offset = self.radius / 111.0
            lon_offset = self.radius / (111.0 * np.cos(np.radians(self.lat)))
            circle_lats.append(self.lat + lat_offset * np.sin(angle))
            circle_lons.append(self.lon + lon_offset * np.cos(angle))
        
        self.circle.set_data(circle_lons, circle_lats)
        self.coord_label.setText(f"Lat: {self.lat:.4f}, Lon: {self.lon:.4f}")

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1 and event.key not in ('control', 'ctrl'):
            self.lat = event.ydata
            self.lon = event.xdata
            self.update_overlays()
            self.canvas.draw_idle()

    def update_radius(self, value):
        self.radius = value
        self.update_overlays()
        self.canvas.draw_idle()

    def accept(self):
        if self.parent:
            self.parent.lat_input.setText(f"{self.lat:.4f}")
            self.parent.lon_input.setText(f"{self.lon:.4f}")
            self.parent.radius_input.setText(str(self.radius))
        super().accept()

    def on_mouse_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3 or (event.button == 1 and event.key in ('control', 'ctrl')):
            self._is_panning = True
            self._last_pan_pos = (event.xdata, event.ydata)
            try:
                self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            except Exception: pass

    def on_mouse_release(self, event):
        if self._is_panning:
            self._is_panning = False
            self._last_pan_pos = None
            try:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception: pass
            self.update_timer.start()

    def on_mouse_move(self, event):
        if not self._is_panning or event.inaxes != self.ax: return
        if self._last_pan_pos is None: return
        
        dx = event.xdata - self._last_pan_pos[0]
        dy = event.ydata - self._last_pan_pos[1]
        
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()
        
    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1.0/1.2 if event.button == 'up' else 1.2
        
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        
        mx = event.xdata
        my = event.ydata
        
        new_w = w * scale
        new_h = h * scale
        
        relx = (mx - x0) / w
        rely = (my - y0) / h
        
        new_x0 = mx - relx * new_w
        new_x1 = new_x0 + new_w
        new_y0 = my - rely * new_h
        new_y1 = new_y0 + new_h
        
        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()
        self.update_timer.start()


class EnhancedProgressWidget(QWidget):
    """Enhanced progress widget with detailed status information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setVisible(False)
        self.status_text.setReadOnly(True)
        self.status_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.layout.addWidget(self.status_text)
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        
        # Progress tracking
        self.start_time = None
        self.last_update = None
        self.current_step = 0
        self.total_steps = 0
        self.status_messages = []
        
        # Create database progress tracker
        self.db_tracker = DatabaseProgressTracker(self)
        
    def start_progress(self, total_steps=0, message="Starting operation..."):
        """Start progress tracking"""
        self.start_time = time.time()
        self.last_update = self.start_time
        self.current_step = 0
        self.total_steps = total_steps
        self.status_messages = [f"[{self._format_time()}] {message}"]
        
        if total_steps > 0:
            self.progress_bar.setRange(0, total_steps)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.progress_bar.setVisible(True)
        self.status_text.setVisible(True)
        self._update_display()
        self.timer.start(100)  # Update every 100ms
        
    def update_progress(self, step=None, message=None):
        """Update progress"""
        current_time = time.time()
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if message:
            self.status_messages.append(f"[{self._format_time()}] {message}")
            # Keep only last 10 messages
            if len(self.status_messages) > 10:
                self.status_messages = self.status_messages[-10:]
        
        self.last_update = current_time
        self._update_display()
        
    def finish_progress(self, message="Operation completed"):
        """Finish progress tracking"""
        self.timer.stop()
        self.status_messages.append(f"[{self._format_time()}] {message}")
        self._update_display()
        
        # Hide after a delay
        QTimer.singleShot(2000, self.hide_progress)
        
    def hide_progress(self):
        """Hide progress widget"""
        self.progress_bar.setVisible(False)
        self.status_text.setVisible(False)
        
    def _format_time(self):
        """Format current time"""
        return datetime.now().strftime("%H:%M:%S")
        
    def _update_display(self):
        """Update the display"""
        if self.total_steps > 0:
            self.progress_bar.setValue(self.current_step)
            
        # Update status text
        status_display = "\n".join(self.status_messages)
        
        # Add timing information
        if self.start_time:
            elapsed = time.time() - self.start_time
            if self.total_steps > 0 and self.current_step > 0:
                eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                status_display += f"\n[{self._format_time()}] Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
            else:
                status_display += f"\n[{self._format_time()}] Elapsed: {elapsed:.1f}s"
                
        self.status_text.setPlainText(status_display)
        
    def update_status(self):
        """Timer callback for status updates"""
        if self.start_time:
            self._update_display()

def check_environment():
    """Check if the environment is correctly set up, return error message if not"""
    errors = []

    # Check Python interpreter
    expected_python = "/home/alan/anaconda3/envs/inat_env/bin/python"
    current_python = sys.executable
    if current_python != expected_python:
        errors.append(
            f"Wrong Python interpreter: {current_python}\n"
            f"Expected: {expected_python}\n"
            "Ensure you activate the 'inat_env' environment with 'conda activate inat_env'."
        )

    # Check for PyQt5 conflict
    if 'PyQt5' in sys.modules:
        errors.append("PyQt5 detected in sys.modules. Uninstall PyQt5 to avoid conflicts with PyQt6.")

    # Check matplotlib backend
    if matplotlib.get_backend() != 'QtAgg':
        errors.append(
            f"Unexpected matplotlib backend: {matplotlib.get_backend()}\n"
            "Expected: QtAgg\n"
            "Ensure matplotlib is configured to use the QtAgg backend."
        )

    # Check required packages
    required_packages = {
        'numpy': '1.26.4',
        'pandas': '2.2.2',
        'pyarrow': '17.0.0',
        'matplotlib': '3.9.2',
        'pyinaturalist': '0.19.0',
        'PyQt6': '6.8.1',
        'duckdb': None  # Version not strictly enforced
    }

    for pkg, expected_version in required_packages.items():
        try:
            module = importlib.import_module(pkg)
            if expected_version and hasattr(module, '__version__'):
                if module.__version__ != expected_version:
                    errors.append(
                        f"{pkg} version mismatch: {module.__version__}\n"
                        f"Expected: {expected_version}"
                    )
        except ImportError:
            errors.append(f"Missing package: {pkg}")

    if errors:
        error_message = (
            "Environment setup issues detected:\n\n" +
            "\n\n".join(errors) +
            "\n\nTo set up the environment correctly, run:\n"
            "```bash\n"
            "conda deactivate\n"
            "conda env remove -n inat_env\n"
            "conda create -n inat_env python=3.12\n"
            "conda activate inat_env\n"
            "conda install numpy=1.26.4 pandas=2.2.2 pyarrow=17.0.0 matplotlib=3.9.2 pyinaturalist=0.19.0\n"
            "pip install PyQt6==6.8.1 duckdb\n"
            "```\n\n"
            "After fixing the environment, restart the program."
        )
        return error_message
    return None

class INatSeasonalVisualizer(QMainWindow):
    """Main application class for iNaturalist Seasonal Visualizer"""
    
    def get_most_recent_date(self):
        """Get the most recent date from the observations.parquet file."""
        if not os.path.exists(self.observations_file):
            return datetime.now().strftime("%Y-%m-%d")

        try:
            con = duckdb.connect()
            schema = con.execute(f"DESCRIBE SELECT * FROM '{self.observations_file}'").fetchall()
            column_names = [row[0].lower() for row in schema]
            if 'eventdate' not in column_names:
                logging.warning("eventDate column not found in observations.parquet. Cannot get most recent date.")
                return datetime.now().strftime("%Y-%m-%d")

            result = con.execute(f"SELECT MAX(eventDate) FROM '{self.observations_file}'").fetchone()
            if result and result[0]:
                # The date might be a datetime object, so we format it
                return pd.to_datetime(result[0]).strftime('%Y-%m-%d')
            else:
                return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            logging.error(f"Failed to get most recent date from database: {str(e)}")
            return datetime.now().strftime("%Y-%m-%d")
        finally:
            if 'con' in locals():
                con.close()

    def __init__(self, lat=None, lon=None, radius=None, scale_factor=1.0, splash_screen=None):
        super().__init__()
        self.splash_screen = splash_screen
        self.settings = QSettings("xAI", "iNatSeasonalVisualizer")
        self.scale_factor = scale_factor
        self.base_font_size = 14  # Centralized base font size in points
        self.api_call_count = 0  # Initialize API call counter
        # In-memory cache for taxon IDs and descendants
        # Persisted to taxon_cache.json to minimize API calls and avoid recomputing descendants
        self.taxon_cache = {}
        self.working_dir = os.getcwd()
        self.taxon_cache_file = os.path.join(self.working_dir, "taxon_cache.json")
        self.descendant_taxons_file = os.path.join(self.working_dir, "descendant_taxons.txt")
        self.taxonomy_file = os.path.join(self.working_dir, "taxonomy.parquet")
        self.observations_file = os.path.join(self.working_dir, "observations.parquet")
        
        # Initialize statusBar and enhanced progress widget early for download_missing_files
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.enhanced_progress = EnhancedProgressWidget()
        
        # Add enhanced progress widget to a temporary layout for visibility during downloads
        self.temp_widget = QWidget()
        self.temp_layout = QVBoxLayout(self.temp_widget)
        self.temp_layout.addWidget(self.enhanced_progress)
        self.setCentralWidget(self.temp_widget)
        
        if self.splash_screen:
            self.splash_screen.update_status("Loading taxon cache...")
            QApplication.processEvents()
        self.load_taxon_cache()
        
        if self.splash_screen:
            self.splash_screen.update_status("Checking for missing data files...")
            QApplication.processEvents()
        self.download_missing_files()  # Check and download missing files
        
        if self.splash_screen:
            self.splash_screen.update_status("Initializing application settings...")
            QApplication.processEvents()
        self.init_args(lat, lon, radius)
        
        # Load database stats for display on startup
        self.total_observations = 0
        self.unique_taxa = 0
        if self.splash_screen:
            self.splash_screen.update_status("Loading database statistics...")
            QApplication.processEvents()
        self.load_database_stats()
        self.most_recent_date = self.get_most_recent_date()

        if self.splash_screen:
            self.splash_screen.update_status("Building user interface...")
            QApplication.processEvents()
        self.init_ui()
        
        if self.splash_screen:
            self.splash_screen.update_status("Loading user preferences...")
            QApplication.processEvents()
        self.load_settings()
        self.update_status_bar()
        self.update_api_call_count()
        
        if self.splash_screen:
            self.splash_screen.update_status("Preparing welcome screen...")
            QApplication.processEvents()
        self.show_placeholder()

    def load_database_stats(self):
        """Query the parquet file to get total observations and unique taxa."""
        if not os.path.exists(self.observations_file):
            return

        try:
            con = duckdb.connect()
            # Check if taxonID column exists before running the main query
            schema = con.execute(f"DESCRIBE SELECT * FROM '{self.observations_file}'").fetchall()
            column_names = [row[0].lower() for row in schema]
            if 'taxonid' not in column_names:
                logging.warning("taxonID column not found in observations.parquet. Cannot calculate unique taxa.")
                self.total_observations = con.execute(f"SELECT COUNT(*) FROM '{self.observations_file}'").fetchone()[0]
                self.unique_taxa = "N/A"
                return

            result = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT taxonID) FROM '{self.observations_file}'").fetchone()
            self.total_observations = result[0]
            self.unique_taxa = result[1]
            logging.info(f"Loaded database stats: {self.total_observations} observations, {self.unique_taxa} unique taxa.")
        except Exception as e:
            logging.error(f"Failed to load database stats: {str(e)}")
            self.total_observations = "Error"
            self.unique_taxa = "Error"
        finally:
            if 'con' in locals():
                con.close()

    def human_readable_size(self, size_bytes):
        """Convert file size in bytes to human-readable format (e.g., KB, MB, GB)"""
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = size_bytes
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.2f} {units[i]}"

    def download_missing_files(self):
        """Check for missing parquet files and prompt user to download them"""
        files_to_download = []
        file_info = {
            "observations.parquet": {
                "url": "http://images.mushroomobserver.org/observations.parquet",
                "size": 1025327222,  # 1.02 GB
                "description": (
                    "observations.parquet (1.02 GB): Contains iNaturalist observation data, including dates, "
                    "locations, and taxon IDs. It enables fast, offline searches to visualize seasonal patterns "
                    "(e.g., Agaricales observations in a region). Without it, searches rely on slower, rate-limited API calls."
                )
            },
            "taxonomy.parquet": {
                "url": "http://images.mushroomobserver.org/taxonomy.parquet",
                "size": 8697166,  # 8.70 MB
                "description": (
                    "taxonomy.parquet (8.70 MB): Contains the iNaturalist taxonomy hierarchy, mapping taxon IDs to their parents. "
                    "It’s essential for resolving hierarchical relationships (e.g., finding all species under Agaricales). "
                    "Without it, searches for higher-level taxa are limited to single taxon IDs."
                )
            }
        }

        # Check which files are missing
        for filename, info in file_info.items():
            file_path = os.path.join(self.working_dir, filename)
            if not os.path.exists(file_path):
                files_to_download.append((filename, info))

        if not files_to_download:
            return  # All files present, no action needed

        # Build message for user
        message = "The following required files are missing and will be downloaded:\n\n"
        for filename, info in files_to_download:
            message += f"- {info['description']}\n\n"
        message += "Do you want to download these files now? Select 'Cancel' to exit the application."

        # Show dialog with Download and Cancel options
        dialog = QMessageBox()
        dialog.setWindowTitle("Missing Files")
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(QMessageBox.StandardButton.Ok)
        dialog.button(QMessageBox.StandardButton.Ok).setText("Download")
        result = dialog.exec()

        if result == QMessageBox.StandardButton.Cancel:
            logging.info("User canceled file download. Exiting application.")
            sys.exit(0)

        # Download each missing file
        for filename, info in files_to_download:
            file_path = os.path.join(self.working_dir, filename)
            url = info["url"]
            human_size = self.human_readable_size(info["size"])
            logging.info(f"Downloading {filename} ({human_size}) from {url}")
            self.statusBar.showMessage(f"Downloading {filename} ({human_size})...")
            
            # Start enhanced progress tracking
            self.enhanced_progress.start_progress(100, f"Downloading {filename} ({human_size})")
            QApplication.processEvents()  # Ensure UI updates

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", info["size"]))
                block_size = 8192
                downloaded = 0

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = min(100, int((downloaded / total_size) * 100))
                                self.enhanced_progress.update_progress(progress, f"Downloaded {self.human_readable_size(downloaded)} / {human_size}")
                                QApplication.processEvents()  # Update progress bar in real-time

                logging.info(f"Successfully downloaded {filename} to {file_path}")
                self.statusBar.showMessage(f"Downloaded {filename} ({human_size})")
                self.enhanced_progress.finish_progress(f"Successfully downloaded {filename}")
                QApplication.processEvents()

            except Exception as e:
                logging.error(f"Failed to download {filename}: {str(e)}")
                self.enhanced_progress.hide_progress()
                QMessageBox.critical(
                    self,
                    "Download Error",
                    f"Failed to download {filename}: {str(e)}\n\n"
                    f"Please manually download it from {url} and place it in {self.working_dir}, "
                    "or ensure your internet connection is stable and try again."
                )
                sys.exit(1)

            finally:
                QApplication.processEvents()

    def init_args(self, lat, lon, radius):
        """Initialize command-line arguments"""
        default_lat = lat if lat is not None else self.settings.value("latitude", 37.7749)
        default_lon = lon if lon is not None else self.settings.value("longitude", -122.4194)
        self.default_lat = f"{float(default_lat):.4f}"
        self.default_lon = f"{float(default_lon):.4f}"
        self.default_radius = radius if radius is not None else self.settings.value("radius", 10)

    def load_taxon_cache(self):
        """Load taxon cache from JSON file to avoid repeated API calls and taxonomy queries"""
        try:
            if os.path.exists(self.taxon_cache_file):
                with open(self.taxon_cache_file, 'r') as f:
                    self.taxon_cache = json.load(f)
                logging.info(f"Loaded taxon cache from {self.taxon_cache_file}")
                for key, value in self.taxon_cache.items():
                    if key.endswith("_descendants"):
                        logging.info(f"Loaded {len(value)} descendant taxon IDs for {key[:-11]}")
            else:
                logging.info(f"No taxon cache found at {self.taxon_cache_file}")
        except Exception as e:
            logging.error(f"Failed to load taxon cache: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Invalid or corrupted taxon_cache.json in {self.working_dir}. Starting with empty cache.")
            self.taxon_cache = {}

    def save_taxon_cache(self):
        """Save taxon cache to JSON file for persistence across sessions"""
        try:
            with open(self.taxon_cache_file, 'w') as f:
                json.dump(self.taxon_cache, f, indent=2)
            logging.info(f"Saved taxon cache to {self.taxon_cache_file}")
        except Exception as e:
            logging.error(f"Failed to save taxon cache: {str(e)}")

    def load_descendant_taxons_from_file(self, query):
        """Load descendant taxon IDs from a user-provided file"""
        try:
            if not os.path.exists(self.descendant_taxons_file):
                return None
            with open(self.descendant_taxons_file, 'r') as f:
                for line in f:
                    if line.strip().startswith(f"{query}:"):
                        taxon_ids = [int(id.strip()) for id in line.split(":")[1].split(",") if id.strip()]
                        logging.info(f"Loaded {len(taxon_ids)} descendant taxon IDs for {query} from {self.descendant_taxons_file}")
                        return taxon_ids
            return None
        except Exception as e:
            logging.error(f"Failed to load descendant taxons from {self.descendant_taxons_file}: {str(e)}")
            return None

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("iNaturalist Seasonal Visualizer")

        # Use the user-provided or default scale factor
        scale_factor = self.scale_factor

        # The window will be launched in a maximized state, so a hardcoded size is not needed.

        # The application font is now set via stylesheet in the toggle_theme method for consistency.

        # Scale matplotlib fonts globally
        matplotlib.rcParams.update({
            'font.size': self.base_font_size * scale_factor,
            'axes.labelsize': self.base_font_size * scale_factor,
            'axes.titlesize': (self.base_font_size + 2) * scale_factor,  # Title is slightly larger
            'xtick.labelsize': (self.base_font_size - 1) * scale_factor,  # Ticks are slightly smaller
            'ytick.labelsize': (self.base_font_size - 1) * scale_factor,
            'legend.fontsize': self.base_font_size * scale_factor,
        })

        # Central widget and main vertical layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Horizontal layout for sidebar and graph
        self.top_layout = QHBoxLayout()

        # Sidebar for inputs
        self.sidebar = QWidget()
        self.sidebar_layout = QFormLayout(self.sidebar)
        self.top_layout.addWidget(self.sidebar, 1)

        # Input fields
        self.lat_input = QLineEdit(self.default_lat)
        self.lat_input.setPlaceholderText("e.g., 37.7749")
        self.lat_input.setMinimumWidth(100)
        self.lat_input.textChanged.connect(self.parse_coordinate_input)

        self.lon_input = QLineEdit(self.default_lon)
        self.lon_input.setPlaceholderText("e.g., -122.4194")
        self.lon_input.setMinimumWidth(100)
        self.lon_input.textChanged.connect(self.parse_coordinate_input)

        self.radius_input = QLineEdit(str(self.default_radius))
        self.organism_input = QLineEdit()
        self.organism_input.setPlaceholderText("e.g., Boletus")
        self.organism_input.returnPressed.connect(self.local_search)
        self.exclude_input = QLineEdit()
        self.exclude_input.setPlaceholderText("e.g., Boletus regineus")
        self.date_from = QLineEdit("2000-01-01")
        self.date_to = QLineEdit(self.most_recent_date)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Daily", "Weekly", "Monthly"])
        self.view_combo.setCurrentIndex(1)  # Set default to Weekly
        self.color_input = QLineEdit(self.settings.value("graph_color", "#1f77b4"))
        self.color_input.setPlaceholderText("e.g., #1f77b4 or blue")
        self.bg_color_input = QLineEdit(self.settings.value("graph_bg_color", ""))
        self.bg_color_input.setPlaceholderText("Theme Default")

        # Create latitude input layout with map button
        lat_layout = QHBoxLayout()
        lat_layout.addWidget(self.lat_input, 2)
        
        # Map button
        self.map_button = QPushButton("🗺️ Map")
        self.map_button.clicked.connect(self.open_map_dialog)
        self.map_button.setToolTip("Open interactive map to set location")
        lat_layout.addWidget(self.map_button, 1)

        # Add latitude layout to sidebar
        self.sidebar_layout.addRow("Latitude:", lat_layout)
        self.sidebar_layout.addRow("Longitude:", self.lon_input)
        self.sidebar_layout.addRow("Radius (km):", self.radius_input)
        self.sidebar_layout.addRow("Organism:", self.organism_input)
        self.sidebar_layout.addRow("Exclude:", self.exclude_input)
        self.sidebar_layout.addRow("Date From:", self.date_from)
        self.sidebar_layout.addRow("Date To:", self.date_to)
        self.sidebar_layout.addRow("View:", self.view_combo)
        self.sidebar_layout.addRow("Graph Color:", self.color_input)
        self.sidebar_layout.addRow("Graph BG Color:", self.bg_color_input)

        # Local Search button
        self.local_search_button = QPushButton("Local Search")
        self.local_search_button.clicked.connect(self.local_search)
        self.sidebar_layout.addWidget(self.local_search_button)

        # Search with API button
        self.search_button = QPushButton("Search with API")
        self.search_button.clicked.connect(self.search_observations)
        self.sidebar_layout.addWidget(self.search_button)

        # Fetch Taxon IDs button
        self.fetch_taxon_ids_button = QPushButton("Fetch Taxon IDs")
        self.fetch_taxon_ids_button.clicked.connect(self.fetch_taxon_ids)
        self.sidebar_layout.addWidget(self.fetch_taxon_ids_button)

        # Show URL button
        self.show_url_button = QPushButton("Show URL")
        self.show_url_button.clicked.connect(self.show_search_url)
        self.sidebar_layout.addWidget(self.show_url_button)

        # Enhanced progress widget for API, local searches, and downloads
        self.sidebar_layout.addWidget(self.enhanced_progress)

        # History list
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_history_item)
        self.sidebar_layout.addRow("History:", self.history_list)

        # Keep the sidebar from expanding past a reasonable width
        avail = QGuiApplication.primaryScreen().availableGeometry()
        self.sidebar.setMaximumWidth(int(avail.width() * 0.35))  # tweak 0.30–0.45 as you like

        # Prevent long history strings from pushing size hints wider
        self.history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # API call count label
        self.api_call_label = QLabel(f"API Calls: {self.api_call_count}")
        self.sidebar_layout.addWidget(self.api_call_label)

        # Graph area
        try:
            self.figure, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.figure)
            self.top_layout.addWidget(self.canvas, 3)

            # Make sure the plot area can shrink instead of forcing the window wider
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.canvas.setMinimumSize(0, 0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize plot: {str(e)}")
            self.canvas = None
            error_label = QLabel(f"Plot initialization failed: {str(e)}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.top_layout.addWidget(error_label, 3)
            return

        # Add top layout to main layout
        self.main_layout.addLayout(self.top_layout)

        # Info bar (moved to bottom)
        self.info_bar = QLabel()
        self.main_layout.addWidget(self.info_bar, stretch=0)

        # Menu bar
        self.init_menu_bar()

        # Status bar (already set in __init__)

        # Apply initial theme
        self.toggle_theme(self.settings.value("theme", "light"))

    def parse_coordinate_input(self, text):
        """Parse coordinate string (e.g. 'lat, lon') and update inputs"""
        if ',' in text:
            parts = text.split(',')
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    
                    # Block signals to prevent recursive updates
                    self.lat_input.blockSignals(True)
                    self.lon_input.blockSignals(True)
                    
                    self.lat_input.setText(f"{lat}")
                    self.lon_input.setText(f"{lon}")
                    
                    self.lat_input.blockSignals(False)
                    self.lon_input.blockSignals(False)
                except ValueError:
                    pass

    def init_menu_bar(self):

        """Initialize the menu bar"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        save_action = QAction("Save Settings", self)
        save_action.triggered.connect(self.save_settings)
        file_menu.addAction(save_action)
        
        export_graph = QAction("Export Graph", self)
        export_graph.triggered.connect(self.export_graph)
        file_menu.addAction(export_graph)
        
        export_data = QAction("Export Data", self)
        export_data.triggered.connect(self.export_data)
        file_menu.addAction(export_data)
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        theme_action = QAction("Toggle Dark/Light Mode", self)
        theme_action.triggered.connect(lambda: self.toggle_theme("dark" if self.settings.value("theme", "light") == "light" else "light"))
        edit_menu.addAction(theme_action)
        
        color_action = QAction("Choose Graph Color", self)
        color_action.triggered.connect(self.choose_color)
        edit_menu.addAction(color_action)

        bg_color_action = QAction("Choose Graph Background Color", self)
        bg_color_action.triggered.connect(self.choose_bg_color)
        edit_menu.addAction(bg_color_action)

        window_bg_action = QAction("Choose Window Background Color", self)
        window_bg_action.triggered.connect(self.choose_window_bg_color)
        edit_menu.addAction(window_bg_action)

    def toggle_theme(self, mode):
        """Toggle between dark and light mode for UI and graph"""
        # Set font size in stylesheet to ensure consistency across theme changes
        font_size_pt = int(self.base_font_size * self.scale_factor)
        font_stylesheet = f"font-size: {font_size_pt}pt;"

        custom_bg_color = self.settings.value("graph_bg_color")
        custom_window_bg = self.settings.value("window_bg_color")
        effective_bg_hex = custom_window_bg if custom_window_bg else ("#ffffff" if mode == "light" else "#2e2e2e")
        contrasting_color = self.get_contrasting_text_color(effective_bg_hex)

        if mode == "dark":
            # UI dark mode
            bg_color = custom_window_bg if custom_window_bg else "#2e2e2e"
            self.setStyleSheet(font_stylesheet + f"background-color: {bg_color}; color: {contrasting_color};")
            self.central_widget.setStyleSheet("QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, QLabel { background-color: #3e3e3e; color: #ffffff; }")
            # Graph dark mode
            if self.canvas:
                graph_contrasting_color = self.get_contrasting_text_color(effective_bg_hex)
                self.figure.set_facecolor(bg_color)
                self.ax.set_facecolor(custom_bg_color if custom_bg_color else "#3e3e3e")
                self.ax.tick_params(colors=graph_contrasting_color)
                self.ax.xaxis.label.set_color(graph_contrasting_color)
                self.ax.yaxis.label.set_color(graph_contrasting_color)
                self.ax.title.set_color(graph_contrasting_color)
                for spine in self.ax.spines.values():
                    spine.set_color(graph_contrasting_color)
        else:
            # UI light mode
            bg_color = custom_window_bg if custom_window_bg else "#ffffff"
            self.setStyleSheet(font_stylesheet + f"background-color: {bg_color}; color: {contrasting_color};")
            self.central_widget.setStyleSheet("QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, QLabel { background-color: #ffffff; color: #000000; }")
            # Graph light mode
            if self.canvas:
                graph_contrasting_color = self.get_contrasting_text_color(effective_bg_hex)
                self.figure.set_facecolor(bg_color)
                self.ax.set_facecolor(custom_bg_color if custom_bg_color else "white")
                self.ax.tick_params(colors=graph_contrasting_color)
                self.ax.xaxis.label.set_color(graph_contrasting_color)
                self.ax.yaxis.label.set_color(graph_contrasting_color)
                self.ax.title.set_color(graph_contrasting_color)
                for spine in self.ax.spines.values():
                    spine.set_color(graph_contrasting_color)
        self.settings.setValue("theme", mode)
        if self.canvas:
            self.canvas.draw_idle()


    def choose_color(self):
        """Open a color dialog to choose the graph color"""
        color = QColorDialog.getColor(QColor(self.color_input.text()), self, "Choose Graph Color")
        if color.isValid():
            self.color_input.setText(color.name())
            self.settings.setValue("graph_color", color.name())
            # Redraw graph if data is loaded
            if self.history_list.property("data"):
                self.search_observations()  # Re-run last search to update color

    def choose_bg_color(self):
        """Open a color dialog to choose the graph background color."""
        current_color_hex = self.settings.value("graph_bg_color", "#ffffff")
        color = QColorDialog.getColor(QColor(current_color_hex), self, "Choose Graph Background Color")
        
        if color.isValid():
            color_name = color.name()
            self.bg_color_input.setText(color_name)
            self.settings.setValue("graph_bg_color", color_name)
            # Re-apply the current theme to update the background
            self.toggle_theme(self.settings.value("theme", "light"))

    def choose_window_bg_color(self):
        """Open a color dialog to choose the main window background color."""
        current_color_hex = self.settings.value("window_bg_color", "#ffffff")
        color = QColorDialog.getColor(QColor(current_color_hex), self, "Choose Window Background Color")
        
        if color.isValid():
            color_name = color.name()
            self.settings.setValue("window_bg_color", color_name)
            # Re-apply the current theme to update the background
            self.toggle_theme(self.settings.value("theme", "light"))

    def open_map_dialog(self):
        """Open the interactive map dialog"""
        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
            radius = float(self.radius_input.text())
        except ValueError:
            # Use defaults if current values are invalid
            lat = float(self.default_lat)
            lon = float(self.default_lon)
            radius = float(self.default_radius)
        
        # Open map dialog
        dialog = MapDialog(self, lat, lon, radius)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Coordinates are updated in the dialog's accept method
            pass

    def load_settings(self):
        """Load saved settings"""
        self.lat_input.setText(self.settings.value("latitude", self.default_lat))
        self.lon_input.setText(self.settings.value("longitude", self.default_lon))
        self.radius_input.setText(self.settings.value("radius", str(self.default_radius)))
        self.color_input.setText(self.settings.value("graph_color", "#1f77b4"))
        self.bg_color_input.setText(self.settings.value("graph_bg_color", ""))

    def save_settings(self):
        """Save current settings"""
        try:
            self.settings.setValue("latitude", float(self.lat_input.text()))
            self.settings.setValue("longitude", float(self.lon_input.text()))
            self.settings.setValue("radius", float(self.radius_input.text()))
            self.settings.setValue("graph_color", self.color_input.text())
            self.settings.setValue("graph_bg_color", self.bg_color_input.text())
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        except ValueError as e:
            QMessageBox.warning(self, "Settings", f"Invalid input: {str(e)}")

    def update_api_call_count(self):
        """Update the API call count display"""
        self.api_call_label.setText(f"API Calls: {self.api_call_count}")

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def fetch_all_observations(self, params):
        """Fetch all observations with pagination, rate limiting, and error handling"""
        all_results = []
        page = 1
        per_page = 500  # Maximum per page for observations endpoint
        max_retries = 3
        rate_limit_delay = 3.0  # Increased for anonymous access

        # Start enhanced progress tracking
        self.enhanced_progress.start_progress(0, "Starting API search...")

        # Use POST for requests with many taxon IDs to avoid 414 Request-URI Too Large
        method = 'get'
        if 'taxon_id' in params and isinstance(params['taxon_id'], list) and len(params['taxon_id']) > 50:
            method = 'post'
            logging.info(f"Using POST request for {len(params['taxon_id'])} taxon IDs to avoid overly long URI.")

        while True:
            params["page"] = page
            params["per_page"] = per_page
            retries = 0

            while retries <= max_retries:
                try:
                    self.statusBar.showMessage(f"Fetching page {page}...")
                    self.enhanced_progress.update_progress(message=f"Fetching page {page}...")
                    response = pyinaturalist.get_observations(**params, method=method)
                    self.api_call_count += 1  # Increment API call counter
                    self.update_api_call_count()
                    results = response.get("results", [])
                    all_results.extend(results)

                    # Update progress bar
                    total_results = response.get("total_results", 0)
                    if total_results > 0:
                        progress = min(100, int((len(all_results) / total_results) * 100))
                        self.enhanced_progress.update_progress(progress, f"Fetched {len(all_results)} / {total_results} observations")

                    # Check if there are more pages
                    if len(all_results) >= total_results or not results:
                        self.enhanced_progress.finish_progress(f"API search completed: {len(all_results)} observations")
                        return all_results, None

                    page += 1
                    time.sleep(rate_limit_delay)  # Rate limiting
                    break  # Exit retry loop on success

                except HTTPError as e:
                    if e.response.status_code in (429, 403):  # Too Many Requests or Forbidden
                        retries += 1
                        wait_time = 2 ** retries  # Exponential backoff: 1s, 2s, 4s
                        self.statusBar.showMessage(f"Error {e.response.status_code}, retrying in {wait_time}s (attempt {retries}/{max_retries})...")
                        self.enhanced_progress.update_progress(message=f"Rate limited, retrying in {wait_time}s (attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        self.enhanced_progress.hide_progress()
                        return all_results, str(e)  # Return partial data with error
                except Exception as e:
                    self.enhanced_progress.hide_progress()
                    return all_results, str(e)  # Return partial data with error

            if retries > max_retries:
                self.enhanced_progress.hide_progress()
                error_msg = (
                    f"Max retries exceeded for error {e.response.status_code}. "
                    "You may be using anonymous API access, which has stricter rate limits. "
                    "To increase limits, set INATURALIST_APP_ID and INATURALIST_APP_SECRET in ~/.bashrc."
                )
                return all_results, error_msg  # Return partial data with error

        self.enhanced_progress.hide_progress()
        return all_results, None

    def get_taxon_id(self, query):
        """Get taxon ID from query, using cache to minimize API calls"""
        if not query:
            return None
        # Check cache to avoid redundant API calls
        if query in self.taxon_cache and isinstance(self.taxon_cache[query], int):
            logging.info(f"Retrieved taxon ID for {query} from cache: {self.taxon_cache[query]}")
            return self.taxon_cache[query]
        try:
            taxa = pyinaturalist.get_taxa(q=query, limit=1)
            self.api_call_count += 1  # Increment API call counter
            self.update_api_call_count()
            if taxa["results"]:
                taxon_id = taxa["results"][0]["id"]
                self.taxon_cache[query] = taxon_id
                self.save_taxon_cache()
                logging.info(f"Fetched taxon ID for {query}: {taxon_id}")
                return taxon_id
            logging.warning(f"No taxon ID found for {query}")
            return None
        except Exception as e:
            error_msg = (
                f"Failed to fetch taxon ID for {query}: {str(e)}. "
                "This may be due to anonymous API access or network issues. "
                "Consider setting INATURALIST_APP_ID and INATURALIST_APP_SECRET in ~/.bashrc for higher limits."
            )
            logging.error(error_msg)
            return None

    def get_descendant_taxon_ids(self, query, taxon_id):
        """Get all descendant taxon IDs for a given taxon using taxonomy.parquet"""
        cache_key = f"{query}_descendants"
        # Check cache to avoid recomputing descendants from taxonomy.parquet
        if cache_key in self.taxon_cache:
            logging.info(f"Retrieved {len(self.taxon_cache[cache_key])} descendant taxon IDs for {query} from cache")
            return self.taxon_cache[cache_key]
        
        # Try loading from user-provided file
        file_taxons = self.load_descendant_taxons_from_file(query)
        if file_taxons:
            self.taxon_cache[cache_key] = file_taxons
            self.save_taxon_cache()
            return file_taxons

        # Query taxonomy.parquet
        try:
            if not os.path.exists(self.taxonomy_file):
                raise FileNotFoundError(f"taxonomy.parquet not found in {self.working_dir}")

            con = duckdb.connect()
            query = """
                WITH RECURSIVE descendants AS (
                    SELECT id FROM taxonomy WHERE id = ?
                    UNION ALL
                    SELECT t.id FROM taxonomy t
                    JOIN descendants d ON t.parent_id = d.id
                )
                SELECT id FROM descendants
            """
            con.execute(f"CREATE VIEW taxonomy AS SELECT * FROM '{self.taxonomy_file}'")
            results = con.execute(query, [taxon_id]).fetchall()
            descendants = [row[0] for row in results]
            
            # Include the parent taxon ID
            if taxon_id not in descendants:
                descendants.append(taxon_id)
            
            self.taxon_cache[cache_key] = descendants
            self.save_taxon_cache()
            logging.info(f"Retrieved {len(descendants)} descendant taxon IDs for {query} from taxonomy.parquet")
            return descendants

        except Exception as e:
            logging.error(f"Failed to fetch descendant taxon IDs for {query} from taxonomy.parquet: {str(e)}")
            
            # Fallback dialog
            error_msg = (
                f"Failed to fetch descendant taxon IDs for {query} from taxonomy.parquet: {str(e)}. "
                "You can:\n"
                f"1. Ensure taxonomy.parquet is in {self.working_dir} and contains valid data.\n"
                f"2. Create {self.descendant_taxons_file} with:\n"
                f"   {query}: taxon_id1, taxon_id2, ... (e.g., Agaricales: 117159, 48723)\n"
                "3. Continue with only the parent taxon ID (may yield fewer results)."
            )
            reply = QMessageBox.question(
                self,
                "Descendant Taxon IDs Failed",
                error_msg + "\n\nContinue with parent taxon ID?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                logging.info(f"Continuing with parent taxon ID {taxon_id} for {query}")
                return [taxon_id]
            else:
                logging.info(f"Aborting descendant taxon ID fetch for {query}")
                return []

    def fetch_taxon_ids(self):
        """Manually fetch taxon IDs for the current organism and save to cache"""
        organism = self.organism_input.text().strip()
        if not organism:
            QMessageBox.warning(self, "Error", "Please enter an organism name.")
            return

        taxon_id = self.get_taxon_id(organism)
        if not taxon_id:
            QMessageBox.critical(self, "Error", f"Could not find taxon ID for {organism}.")
            return

        self.enhanced_progress.start_progress(0, f"Fetching taxon IDs for {organism}...")
        self.statusBar.showMessage(f"Fetching taxon IDs for {organism}...")

        try:
            self.enhanced_progress.update_progress(message="Querying taxonomy database...")
            taxon_ids = self.get_descendant_taxon_ids(organism, taxon_id)
            self.enhanced_progress.finish_progress(f"Fetched {len(taxon_ids)} taxon IDs for {organism}")
            if taxon_ids:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Fetched {len(taxon_ids)} taxon IDs for {organism}. Saved to {self.taxon_cache_file}."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"No taxon IDs fetched for {organism}. Check logs or try again."
                )
        except Exception as e:
            self.enhanced_progress.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to fetch taxon IDs: {str(e)}")
        finally:
            self.statusBar.showMessage("Ready")

    def show_search_url(self):
        """Show the iNaturalist URL for the current search parameters"""
        try:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid latitude or longitude. Please enter valid numbers.")
                return

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()

            # Fetch taxon ID for organism
            taxon_id = self.get_taxon_id(organism)

            # Construct URL
            params = {
                "lat": lat,
                "lng": lon,
                "radius": radius,
                "d1": date_from,
                "d2": date_to,
                "subview": "map"
            }
            if taxon_id:
                params["taxon_id"] = taxon_id

            base_url = "https://www.inaturalist.org/observations"
            url = f"{base_url}?{urlencode(params)}"
            QMessageBox.information(self, "iNaturalist URL", f"Search URL:\n{url}\n\nCopy this URL to verify the search on iNaturalist.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate URL: {str(e)}")

    def clamp_to_screen(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()

        g = self.frameGeometry()
        w = min(g.width(), avail.width())
        h = min(g.height(), avail.height())

        if g.width() != w or g.height() != h:
            self.resize(w, h)

        # keep top-left inside the available area
        x = max(avail.left(), min(self.x(), avail.right() - w))
        y = max(avail.top(),  min(self.y(), avail.bottom() - h))
        self.move(x, y)

    def plot_observations(self, observations, lat, lon, radius, organism, date_from, date_to, view, source="API"):
        """Plot observations (from API or local)"""
        if not observations:
            self.show_placeholder(f"No {source} observations found.")
            return False

        # Process data, handling timezone-aware timestamps
        dates = []
        for obs in observations:
            date_str = obs.get("eventDate" if source == "Local" else "observed_on")
            if date_str:
                try:
                    # Convert to UTC and make timezone-naive
                    date = pd.to_datetime(date_str, utc=True).tz_localize(None)
                    dates.append(date)
                except Exception as e:
                    logging.warning(f"Skipping invalid date at observation {obs}: {str(e)}")
                    continue
            else:
                logging.warning(f"Skipping observation with missing date: {obs}")
                continue

        if not dates:
            self.show_placeholder(f"No valid observation dates found in {source} data.")
            return False

        df = pd.DataFrame(dates, columns=["date"])
        
        if view == "daily":
            df["group"] = df["date"].dt.dayofyear
            bins = 366  # Account for leap years
            labels = [f"Day {i}" for i in range(1, 367)]
            tick_interval = 30
            tick_positions = range(0, bins, tick_interval)
        elif view == "weekly":
            df["group"] = df["date"].dt.isocalendar().week
            bins = 52
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            # Evenly space 12 months across 52 weeks (0 to 51)
            tick_positions = np.linspace(0, 51, 12, endpoint=True)
            tick_interval = None
        else:  # monthly
            df["group"] = df["date"].dt.month
            bins = 12
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            tick_interval = 1
            tick_positions = range(bins)

        counts = df["group"].value_counts().sort_index()
        data = np.zeros(bins)
        for idx, count in counts.items():
            index = int(idx) - 1
            if 0 <= index < bins:
                data[index] = count
            else:
                logging.warning(f"Skipping out-of-bounds index: {index}")

        # Update graph
        self.ax.clear()
        self.ax.bar(range(bins), data, color=self.color_input.text())
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(labels, rotation=45)
        self.ax.set_xlabel("Time of Year")
        self.ax.set_ylabel("Observation Count")
        self.ax.set_title(f"Seasonal Observations for {organism or 'All Organisms'} within {radius} km of ({lat:.3f}, {lon:.3f})")
        
        # Apply theme to graph
        current_theme = self.settings.value("theme", "light")
        self.toggle_theme(current_theme)  # Reapply theme to update graph colors
        
        self.canvas.draw_idle()
        QTimer.singleShot(0, self.clamp_to_screen)

        # Update history
        history_item = f"{organism or 'All'} | {date_from} to {date_to} ({source})"
        self.history_list.addItem(history_item)
        self.history_list.setProperty("data", observations)

        # Update info bar with observation count and descendant taxa
        observation_count = len(observations)
        descendant_count = 0
        if organism:
            cache_key = f"{organism}_descendants"
            if cache_key in self.taxon_cache:
                descendant_count = len(self.taxon_cache[cache_key])
        descendant_text = f", Descendant Taxa: {descendant_count}" if descendant_count > 0 else ""
        self.info_bar.setText(
            f"Source: {source}, Location: ({lat}, {lon}), Radius: {radius} km, "
            f"Organism: {organism or 'All'}, Dates: {date_from} to {date_to}, "
            f"Observations: {observation_count}{descendant_text}"
        )
        self.update_status_bar(observation_count)
        return True

    def local_search(self):
        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            exclude = self.exclude_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()

            # Check for parquet file in current working directory
            parquet_path = self.observations_file
            if not os.path.exists(parquet_path):
                QMessageBox.critical(
                    self,
                    "Error",
                    f"observations.parquet not found in {self.working_dir}.\n"
                    "Please ensure the file is in the current working directory."
                )
                return

            # Connect to DuckDB
            con = duckdb.connect()

            # Register the haversine function
            con.create_function("haversine", self.haversine, [float, float, float, float], float)

            # Check schema
            schema = con.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").fetchall()
            column_names = [row[0].lower() for row in schema]
            required_columns = ["eventdate", "decimallatitude", "decimallongitude", "taxonid"]
            missing_columns = [col for col in required_columns if col not in column_names]
            if missing_columns:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Missing required columns in observations.parquet: {', '.join(missing_columns)}.\n\n"
                    "The file must include 'eventDate', 'decimalLatitude', 'decimalLongitude', and 'taxonID'.\n"
                    "Please regenerate the parquet file with these columns. For example, if using a CSV:\n"
                    "COPY (\n"
                    "  SELECT id, decimalLatitude, decimalLongitude, eventDate, taxonID\n"
                    "  FROM read_csv_auto('observations.csv')\n"
                    ") TO 'observations.parquet' (FORMAT PARQUET);\n"
                )
                return

            # Bounding box calculation
            lat_rad = math.radians(lat)
            # Radius of Earth in kilometers
            R = 6371
            lat_diff = radius / R

            # Prevent math domain error for large radii near poles
            asin_arg = math.sin(lat_diff) / math.cos(lat_rad)
            if asin_arg > 1.0:
                asin_arg = 1.0
            elif asin_arg < -1.0:
                asin_arg = -1.0
            lon_diff = math.asin(asin_arg)

            lat_min = lat - math.degrees(lat_diff)
            lat_max = lat + math.degrees(lat_diff)
            lon_min = lon - math.degrees(lon_diff)
            lon_max = lon + math.degrees(lon_diff)

            # Build DuckDB query
            query = f"""
                SELECT id, decimalLatitude, decimalLongitude, eventDate, taxonID
                FROM '{parquet_path.replace("'", "''")}'
                WHERE eventDate BETWEEN ? AND ?
                AND decimalLatitude BETWEEN ? AND ?
                AND decimalLongitude BETWEEN ? AND ?
                AND haversine(?, ?, decimalLatitude, decimalLongitude) <= ?
            """
            params = [date_from, date_to, lat_min, lat_max, lon_min, lon_max, lat, lon, radius]

            # Add taxon filter if organism is specified
            taxon_ids = []
            if organism:
                taxon_id = self.get_taxon_id(organism)
                if taxon_id:
                    taxon_ids = self.get_descendant_taxon_ids(organism, taxon_id)
                    if taxon_ids:
                        query += " AND taxonID IN (" + ",".join([str(id) for id in taxon_ids]) + ")"
                    else:
                        QMessageBox.warning(self, "Warning", f"No taxon IDs found for {organism}. Search may yield no results.")
                else:
                    QMessageBox.warning(self, "Warning", f"Could not find taxon ID for {organism}. Search may yield no results.")

            # Add exclude filter
            exclude_taxon_ids = []
            if exclude:
                exclude_taxon_id = self.get_taxon_id(exclude)
                if exclude_taxon_id:
                    exclude_taxon_ids = self.get_descendant_taxon_ids(exclude, exclude_taxon_id)
                    if exclude_taxon_ids:
                        query += " AND taxonID NOT IN (" + ",".join([str(id) for id in exclude_taxon_ids]) + ")"

            # Start database progress tracking with estimated progress
            self.enhanced_progress.db_tracker.start_operation("local database search")
            self.statusBar.showMessage("Performing local search...")
            QApplication.processEvents()

            # Get estimated count for better progress tracking
            self.enhanced_progress.db_tracker.update_progress(message="Estimating result count...")
            count_query = f"""
                SELECT COUNT(*) FROM '{parquet_path.replace("'", "''")}'
                WHERE eventDate BETWEEN ? AND ?
                AND decimalLatitude BETWEEN ? AND ?
                AND decimalLongitude BETWEEN ? AND ?
            """
            count_params = [date_from, date_to, lat_min, lat_max, lon_min, lon_max]
            if taxon_ids:
                count_query += " AND taxonID IN (" + ",".join([str(id) for id in taxon_ids]) + ")"
            if exclude_taxon_ids:
                count_query += " AND taxonID NOT IN (" + ",".join([str(id) for id in exclude_taxon_ids]) + ")"
            
            estimated_count = con.execute(count_query, count_params).fetchone()[0]
            self.enhanced_progress.db_tracker.estimated_total = estimated_count
            
            # Execute main query with progress updates
            self.enhanced_progress.db_tracker.update_progress(message=f"Executing DuckDB query (estimated {estimated_count:,} results)...")
            results = con.execute(query, params).fetchall()
            
            self.enhanced_progress.db_tracker.update_progress(message="Processing results...")
            # Convert results to list of dicts
            observations = [
                {
                    "id": row[0],
                    "decimalLatitude": row[1],
                    "decimalLongitude": row[2],
                    "eventDate": row[3],
                    "taxonID": row[4],
                }
                for row in results
            ]

            self.enhanced_progress.db_tracker.finish_operation(len(observations), "Local search completed")
            self.statusBar.showMessage("Local search completed.")

            # Plot results
            if observations:
                self.plot_observations(
                    observations, lat, lon, radius, organism, date_from, date_to, view, source="Local"
                )
            else:
                self.show_placeholder("No local observations found.")
                self.update_status_bar(0)

        except Exception as e:
            self.enhanced_progress.hide_progress()
            QMessageBox.critical(self, "Error", f"Local search failed: {str(e)}")
            self.show_placeholder(f"Local search failed: {str(e)}")
            logging.error(f"Local search failed: {str(e)}")
        finally:
            if 'con' in locals():
                con.close()

    def search_observations(self):
        """Search observations using iNaturalist API"""
        if not self.canvas:
            QMessageBox.critical(self, "Error", "Cannot search: Plot initialization failed.")
            return

        try:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid latitude or longitude. Please enter valid numbers.")
                return

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            exclude = self.exclude_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()

            # Build API parameters
            params = {
                "lat": lat,
                "lng": lon,
                "radius": radius,
                "d1": date_from,
                "d2": date_to,
                "per_page": 500
            }

            # Add taxon filter if organism is specified
            taxon_ids = []
            if organism:
                taxon_id = self.get_taxon_id(organism)
                if taxon_id:
                    taxon_ids = self.get_descendant_taxon_ids(organism, taxon_id)
                    if taxon_ids:
                        params["taxon_id"] = taxon_ids
                    else:
                        QMessageBox.warning(self, "Warning", f"No taxon IDs found for {organism}. Search may yield no results.")
                else:
                    QMessageBox.warning(self, "Warning", f"Could not find taxon ID for {organism}. Search may yield no results.")

            # Add exclude filter
            if exclude:
                exclude_taxon_id = self.get_taxon_id(exclude)
                if exclude_taxon_id:
                    exclude_taxon_ids = self.get_descendant_taxon_ids(exclude, exclude_taxon_id)
                    if exclude_taxon_ids:
                        params["not_in_taxon_id"] = exclude_taxon_ids

            # Fetch observations
            observations, error = self.fetch_all_observations(params)
            if error:
                QMessageBox.critical(self, "Error", f"API search failed: {error}")
                self.show_placeholder(f"API search failed: {error}")
                return

            # Plot results
            if observations:
                self.plot_observations(
                    observations, lat, lon, radius, organism, date_from, date_to, view, source="API"
                )
            else:
                self.show_placeholder("No API observations found.")
                self.update_status_bar(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"API search failed: {str(e)}")
            self.show_placeholder(f"API search failed: {str(e)}")
            logging.error(f"API search failed: {str(e)}")

    def export_graph(self):
        """Export the current graph as an image with metadata"""
        if not self.canvas:
            QMessageBox.critical(self, "Error", "No graph to export: Plot initialization failed.")
            return

        try:
            # Get current parameters
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid latitude or longitude. Using default values for export.")
                lat = float(self.default_lat)
                lon = float(self.default_lon)

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip() or "All Organisms"
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()
            source = "Local" if self.info_bar.text().startswith("Source: Local") else "API"
            observation_count = len(self.history_list.property("data") or [])
            descendant_count = 0
            if organism != "All Organisms":
                cache_key = f"{organism}_descendants"
                if cache_key in self.taxon_cache:
                    descendant_count = len(self.taxon_cache[cache_key])

            # Open file dialog
            file_dialog = QFileDialog(self, "Export Graph", "", "JPG (*.jpg);;PNG (*.png)")
            file_dialog.setDefaultSuffix("jpg")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if not file_dialog.exec():
                return

            filename = file_dialog.selectedFiles()[0]
            # Ensure .jpg extension
            file_path = Path(filename)
            if file_path.suffix.lower() not in [".jpg", ".jpeg"]:
                filename = str(file_path.with_suffix(".jpg"))

            # Create a new figure for export to avoid modifying the UI graph
            export_fig, export_ax = plt.subplots(figsize=(10, 6))

            # Recompute plot data (replicate plot_observations logic)
            observations = self.history_list.property("data") or []
            if not observations:
                plt.close(export_fig)
                QMessageBox.warning(self, "Error", "No data to export.")
                return

            dates = []
            for obs in observations:
                date_str = obs.get("eventDate" if source == "Local" else "observed_on")
                if date_str:
                    try:
                        date = pd.to_datetime(date_str, utc=True).tz_localize(None)
                        dates.append(date)
                    except Exception:
                        continue

            if not dates:
                plt.close(export_fig)
                QMessageBox.warning(self, "Error", "No valid dates to export.")
                return

            df = pd.DataFrame(dates, columns=["date"])
            
            if view == "daily":
                df["group"] = df["date"].dt.dayofyear
                bins = 366
                labels = [f"Day {i}" for i in range(1, 367)]
                tick_positions = range(0, bins, 30)
            elif view == "weekly":
                df["group"] = df["date"].dt.isocalendar().week
                bins = 52
                labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                tick_positions = np.linspace(0, 51, 12, endpoint=True)
            else:  # monthly
                df["group"] = df["date"].dt.month
                bins = 12
                labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                tick_positions = range(bins)

            counts = df["group"].value_counts().sort_index()
            data = np.zeros(bins)
            for idx, count in counts.items():
                index = int(idx) - 1
                if 0 <= index < bins:
                    data[index] = count

            # Plot on new figure
            export_ax.bar(range(bins), data, color=self.color_input.text())
            export_ax.set_xticks(tick_positions)
            export_ax.set_xticklabels(labels, rotation=45)
            export_ax.set_xlabel("Time of Year")
            export_ax.set_ylabel("Observation Count")
            export_ax.set_title(f"Seasonal Observations for {organism} ({source})")

            # Apply theme to export graph
            current_theme = self.settings.value("theme", "light")
            if current_theme == "dark":
                export_fig.set_facecolor("#2e2e2e")
                export_ax.set_facecolor("#3e3e3e")
                export_ax.tick_params(colors="white")
                export_ax.xaxis.label.set_color("white")
                export_ax.yaxis.label.set_color("white")
                export_ax.title.set_color("white")
                for spine in export_ax.spines.values():
                    spine.set_color("white")
            else:
                export_fig.set_facecolor("white")
                export_ax.set_facecolor("white")
                export_ax.tick_params(colors="black")
                export_ax.xaxis.label.set_color("black")
                export_ax.yaxis.label.set_color("black")
                export_ax.title.set_color("black")
                for spine in export_ax.spines.values():
                    spine.set_color("black")

            # Add metadata text
            metadata = (
                f"Source: {source}\n"
                f"Location: ({lat}, {lon})\n"
                f"Radius: {radius} km\n"
                f"Organism: {organism}\n"
                f"Dates: {date_from} to {date_to}\n"
                f"Observations: {observation_count}\n"
                f"Descendant Taxa: {descendant_count}" if descendant_count > 0 else ""
            )
            export_fig.text(0.02, 0.02, metadata, fontsize=8, transform=export_fig.transFigure, verticalalignment='bottom')

            # Render and save
            export_fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for metadata
            export_fig.canvas.draw()
            export_fig.savefig(filename, format='jpg', dpi=300)
            plt.close(export_fig)

            QMessageBox.information(self, "Success", f"Graph exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export graph: {str(e)}")
            logging.error(f"Export graph failed: {str(e)}")

    def export_data(self):
        """Export observation data as CSV"""
        observations = self.history_list.property("data") or []
        if not observations:
            QMessageBox.warning(self, "Error", "No data to export.")
            return

        try:
            file_dialog = QFileDialog(self, "Export Data", "", "CSV (*.csv)")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if not file_dialog.exec():
                return

            filename = file_dialog.selectedFiles()[0]
            source = "Local" if self.info_bar.text().startswith("Source: Local") else "API"
            data = []
            for obs in observations:
                data.append({
                    "id": obs.get("id"),
                    "latitude": obs.get("decimalLatitude" if source == "Local" else "geojson", {}).get("coordinates", [None, None])[1],
                    "longitude": obs.get("decimalLongitude" if source == "Local" else "geojson", {}).get("coordinates", [None, None])[0],
                    "date": obs.get("eventDate" if source == "Local" else "observed_on"),
                    "taxonID": obs.get("taxonID" if source == "Local" else "taxon_id")
                })

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
            logging.error(f"Export data failed: {str(e)}")

    def load_history_item(self):
        """Load and replot a history item"""
        observations = self.history_list.property("data") or []
        if not observations:
            return

        try:
            location_text = self.location_input.text()
            lat, lon = None, None
            if "," in location_text:
                parts = location_text.split(',')
                if len(parts) == 2:
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                    except ValueError:
                        pass # Handle invalid coordinate format
            if lat is None or lon is None:
                QMessageBox.warning(self, "Warning", "Invalid location coordinates. Using default values for history item.")
                lat = float(self.default_lat)
                lon = float(self.default_lon)

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip() or "All Organisms"
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()
            source = "Local" if self.history_list.currentItem().text().endswith("(Local)") else "API"

            self.plot_observations(
                observations, lat, lon, radius, organism, date_from, date_to, view, source
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load history item: {str(e)}")
            logging.error(f"Load history item failed: {str(e)}")

    def show_placeholder(self, message=None):
        """Show a placeholder message on the graph"""
        if self.canvas:
            # Determine text color based on background luminance for visibility
            current_theme = self.settings.value("theme", "light")
            custom_window_bg = self.settings.value("window_bg_color")
            effective_bg_hex = custom_window_bg if custom_window_bg else ("#ffffff" if current_theme == "light" else "#2e2e2e")
            text_color = self.get_contrasting_text_color(effective_bg_hex)

            self.ax.clear()
            if message:
                display_message = message
            else:
                # Load current settings from UI widgets to display them
                lat = self.lat_input.text()
                lon = self.lon_input.text()

                radius = self.radius_input.text()
                date_from = self.date_from.text()
                date_to = self.date_to.text()
                graph_color = self.color_input.text() or "#1f77b4"
                graph_bg_color = self.bg_color_input.text() or "Theme Default"
                window_bg_color = self.settings.value("window_bg_color", "Theme Default")

                settings_text = (
                    f"Current Settings:\n\n"
                    f"  - Latitude: {lat}\n"
                    f"  - Longitude: {lon}\n"
                    f"  - Radius: {radius} km\n"
                    f"  - Date Range: {date_from} to {date_to}\n"
                    f"  - Graph Color: {graph_color}\n"
                    f"  - Graph Background Color: {graph_bg_color}\n"
                    f"  - Window Background Color: {window_bg_color}\n\n"
                )

                display_message = (
                    "Welcome to iNaturalist Seasonal Visualizer!\n\n" +
                    "To get started:\n"
                    "1. Enter an organism name (e.g., Russula brevipes, Agaricales, etc).\n"
                    "2. Click 'Local Search' to use local data or 'Search with API' for online data.\n\n" +
                    "The graph will display seasonal observation patterns.\n\n" +
                    settings_text
                )

                if self.total_observations:
                    try:
                        obs_count = f"{self.total_observations:,}"
                        taxa_count = f"{self.unique_taxa:,}"
                    except (ValueError, TypeError): # Handle "Error" or "N/A" strings
                        obs_count = self.total_observations
                        taxa_count = self.unique_taxa

                    database_stats_text = (
                        f"Local Database Stats:\n"
                        f"  - Total Observations: {obs_count}\n"
                        f"  - Unique Taxa: {taxa_count}\n\n"
                    )
                    display_message += database_stats_text
            self.ax.text(0.5, 0.5, display_message, color=text_color, ha='center', va='center', wrap=True)
            self.ax.set_axis_off()
            self.canvas.draw()
            self.info_bar.setText("")
            self.update_status_bar(0)

    def update_status_bar(self, observation_count=0):
        """Update the status bar with current info"""
        self.statusBar.showMessage(f"Ready | Observations: {observation_count} | API Calls: {self.api_call_count}")

    def get_contrasting_text_color(self, bg_color_hex, light_fallback='white', dark_fallback='black'):
        """Determines if black or white text has better contrast against a given hex background color."""
        try:
            q_color = QColor(bg_color_hex)
            r, g, b, _ = q_color.getRgb()
            # Using standard luminance formula
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            return dark_fallback if luminance > 128 else light_fallback
        except Exception:
            # Fallback if the color string is invalid
            current_theme = self.settings.value("theme", "light")
            return dark_fallback if current_theme == "light" else light_fallback

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="iNaturalist Seasonal Visualizer")
    parser.add_argument("--lat", type=float, help="Latitude (default from settings)")
    parser.add_argument("--lon", type=float, help="Longitude (default from settings)")
    parser.add_argument("--radius", type=float, help="Radius in km (default from settings)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Manual UI scaling factor (e.g., 2.0 for 200%% scaling)")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename="inat_visualizer.log",
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check environment
    env_error = check_environment()
    if env_error:
        print(env_error)
        sys.exit(1)

    # --- High DPI stability setup (Wayland + Qt6) ---
    QApplication.setHighDpiScaleFactorRoundingPolicy( Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor)

    # Create QApplication first
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash_image_path = os.path.join(os.getcwd(), "splash_screen.jpg")
    if os.path.exists(splash_image_path):
        splash = CustomSplashScreen(splash_image_path)
        splash.update_status("Starting iNaturalist Seasonal Visualizer...")
        QApplication.processEvents()
        time.sleep(0.5)  # Brief pause to show splash screen
    else:
        splash = None
        logging.warning(f"Splash screen image not found at {splash_image_path}")
    
    # Initialize main window with splash screen updates
    if splash:
        splash.update_status("Loading application components...")
        QApplication.processEvents()
        time.sleep(0.3)
    
    window = INatSeasonalVisualizer(lat=args.lat, lon=args.lon, radius=args.radius, scale_factor=args.scale_factor, splash_screen=splash)
    
    if splash:
        splash.update_status("Application ready!")
        QApplication.processEvents()
        time.sleep(0.5)
        splash.close()
    
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
