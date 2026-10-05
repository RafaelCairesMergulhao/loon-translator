# Commercial release checklist

The application is designed for a one-time purchase: no ads, no mandatory
subscription and no per-minute fee. Local recognition, local translation packs
and installed Windows voices support that model. Cloud providers remain
optional conveniences and must never be advertised as permanently free or
guaranteed.

## Required before selling

1. Register the product and publisher name, and obtain legal review of the
   EULA, privacy policy, trademark and third-party model licenses.
2. Audit each downloadable Argos model. Do not enable a model in the commercial
   catalog if its data/model license is missing or prohibits commercial use.
3. Obtain a Windows code-signing certificate and build with
   `LOON_CERT_THUMBPRINT` configured.
4. Test the signed build on clean Windows 10 and Windows 11 machines with
   different microphones, headsets, Bluetooth devices and VB-CABLE versions.
5. Complete accessibility, localization, crash recovery and long-call soak
   testing.
6. Have at least two native speakers review recognition, translation and voice
   quality for every advertised language pair.
7. Publish a support address, refund/support process and an update policy.
8. Do not bundle VB-CABLE without written redistribution permission.

## Steam

1. Create a Steamworks partner account, pay the app fee and obtain AppID and
   DepotID values.
2. Replace placeholders in `distribution/steam/*.vdf`.
3. Configure the launch executable as `LoonTranslator.exe`, Windows 64-bit.
4. Enable the Microsoft Visual C++ common redistributable if required by the
   final frozen build.
5. Complete Valve's Content Survey. The product performs live AI transcription,
   translation and speech synthesis from user-provided content; explain the
   privacy controls and that output is returned only to the user's selected
   audio devices.
6. Upload using SteamPipe from the Steamworks SDK and test on a private beta
   branch before review.
7. The store page must clearly state that a virtual audio driver such as
   VB-CABLE is required for injecting translated speech into third-party apps.

Steamworks access, store submission, tax/banking onboarding and Valve approval
cannot be automated from this repository.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The Steam-ready folder is generated at `dist\LoonTranslator`. Preserve the
generated SHA-256 file with each release.

## Release gates

- Automated tests pass.
- No high-severity security or privacy finding is open.
- Executable is signed and its signature verifies on a clean machine.
- Offline mode works after disconnecting the network.
- Cloud refusal/failure never blocks access to installed local packs.
- Start/stop, device loss and two-hour calls do not leak memory or deadlock.
- All advertised languages have approved model licenses and quality evidence.
