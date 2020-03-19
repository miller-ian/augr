Due to the use of hardware accelerated graphics, ATAK may not be able to run in an AVD emulator in a given host environment.

The emulator needs to run with GLES 3.0 or later compatibility. Once the emulator is launched, click the Menu (`...`) button on the menu bar, select `Settings` then from the `Settings` screen, click the `Advanced` tab. Ensure that the `Open GL ES API Level` is set to `GLES 3.0` or higher. For `Open GL ES renderer`, the various `ANGLE` options should provide good compatibility for Windows environments; use of other options will vary depending on the graphics configuration of the host.  
Once you've made changes, be sure to restart the emulator!!!
