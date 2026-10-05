; Instalador do Loon Translator (Inno Setup 6).
; Gerado por scripts\build_installer.ps1, que passa /DSourceDir com a pasta do PyInstaller.

#define AppName "Loon Translator"
#define AppVersion "0.5.0"
#define AppExe "LoonTranslator.exe"
#define VBCableKey "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VB:VBCABLE {87459874-1236-4469}"
#ifndef SourceDir
  #define SourceDir "..\dist\LoonTranslator"
#endif

[Setup]
AppId={{6F1C2B4E-8D3A-4F5B-9C7E-1A2B3C4D5E6F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Loon Translator
DefaultDirName={autopf}\LoonTranslator
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Instala sem pedir administrador; quem quiser pode escolher "para todos os usuários".
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\release
OutputBaseFilename=LoonTranslator-{#AppVersion}-Setup
SetupIconFile=..\assets\loon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
WizardSizePercent=110
InfoBeforeFile=LEIA-ME.txt
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
CloseApplications=yes
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} - instalador

[Languages]
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"

[CustomMessages]
ptbr.OpenVBCable=Baixar o VB-CABLE (necessário para usar em chamadas)
en.OpenVBCable=Download VB-CABLE (required for calls)
es.OpenVBCable=Descargar VB-CABLE (necesario para llamadas)
ptbr.OpenGuide=Abrir o guia rápido
en.OpenGuide=Open the quick guide
es.OpenGuide=Abrir la guía rápida
ptbr.Guide=Guia rápido
en.Guide=Quick guide
es.Guide=Guía rápida

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "LEIA-ME.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{group}\{cm:Guide}"; Filename: "{app}\LEIA-ME.txt"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "https://vb-audio.com/Cable/"; Description: "{cm:OpenVBCable}"; Flags: postinstall shellexec skipifsilent; Check: NeedsVBCable
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal\__pycache__"

[Code]
function NeedsVBCable: Boolean;
begin
  Result := not RegKeyExists(HKLM64, '{#VBCableKey}') and not RegKeyExists(HKLM32, '{#VBCableKey}');
end;
