# Download and install

The downloads under **Assets** are standalone applications - No Python installation
is required.

> [!IMPORTANT]
> These builds are not signed with a paid Windows code-signing certificate or
> an Apple Developer ID, so Windows and macOS will show a security warning on
> first launch. Only continue if you downloaded the file from the official
> `AlanRockefeller/inat.visualizer.py` release page.

## Windows 10 or 11

1. Under **Assets**, download `iNat-Seasonal-Visualizer-Windows.zip`.
2. Extract the ZIP file, open the extracted `iNat-Seasonal-Visualizer` folder,
   and double-click `iNat-Seasonal-Visualizer.exe`. Keep the other extracted
   files beside the executable; together they make up the portable app.
3. If Microsoft Defender SmartScreen says **Windows protected your PC**, click
   **More info**. Confirm that the app name is
   `iNat-Seasonal-Visualizer.exe`, then click **Run anyway**.

Windows shows this warning because the executable has no publisher certificate;
it does not by itself mean that Windows found malware.

[Microsoft's explanation of SmartScreen warnings for unsigned apps](https://learn.microsoft.com/windows/apps/package-and-deploy/smartscreen-reputation)

## macOS

1. Choose the correct download under **Assets**:
   - Macs whose **Apple menu > About This Mac** window lists an Apple chip (M1,
     M2, M3, M4, or later):
     `iNat-Seasonal-Visualizer-macOS-Apple-Silicon.zip`
   - Macs whose **About This Mac** window lists an Intel processor:
     `iNat-Seasonal-Visualizer-macOS-Intel.zip`
2. Double-click the downloaded ZIP file to extract it, then drag
   `iNat-Seasonal-Visualizer` into your **Applications** folder.
3. Double-click the app once. macOS will say that it cannot verify the developer
   or check the app for malicious software. Close that message.
4. Open **Apple menu > System Settings > Privacy & Security**, scroll down to
   **Security**, and click **Open Anyway** beside the message about
   `iNat-Seasonal-Visualizer`. This option is available for about an hour after
   step 3.
5. Enter your Mac login password if asked, then confirm by clicking **Open**.
   This exception is saved, so later launches work normally.

On older macOS versions, you may instead be able to Control-click the app,
choose **Open**, and then click **Open** again.

[Apple's official instructions for opening an app from an unknown developer](https://support.apple.com/guide/mac-help/mh40616/mac)

## Linux

Download `iNat-Seasonal-Visualizer-Linux.tar.gz`, extract it, open the extracted
`iNat-Seasonal-Visualizer` directory, and run the `iNat-Seasonal-Visualizer`
executable inside it.

## First launch

The app offers to download an optional local database of approximately 1 GB.
The database makes local searches faster. If you skip it, the app still works
and searches through the online iNaturalist API instead.
