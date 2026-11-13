' PostureGuard Silent Launcher
' Launches PostureGuard desktop monitor in the background with zero terminal window.
Set WshShell = CreateObject("WScript.Shell")
strPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strPath
WshShell.Run """" & strPath & "\venv\Scripts\pythonw.exe"" """ & strPath & "\run_desktop.py""", 0, False
