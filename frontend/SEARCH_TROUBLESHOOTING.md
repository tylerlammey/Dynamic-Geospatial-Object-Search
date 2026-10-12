# Search blocked by Windows Application Control (WinError 4551)

For **WinError 5** mentioning a temporary file being renamed to `status.json`,
the frontend previously failed when its progress poller briefly held the old
status file open. The writer now retries Windows errors 5, 32, and 33 for about
five seconds, preserving the old complete JSON until the replacement succeeds.
Persistent errors still propagate. Cache publication also preserves its old
accounting on failure. The fix was tested with a real Windows reader handle,
and is included in both the checkout service and rebuilt Windows package.

The earlier local Code Integrity event log confirmed that Windows blocked the existing
AMD PyTorch `torch/lib/aotriton_v2.dll` itself. Its Authenticode status is
`NotSigned`, and Smart App Control was enabled during that diagnosis
(`VerifiedAndReputablePolicyState=1`). The installed PyTorch build is
`2.9.1+rocm7.2.1`.

This prevents a new Python search process from importing PyTorch. Imagery
browsing and saved-result browsing can continue because they do not load that
GPU library. Rebuilding the frontend or regenerating its photographic overviews
does not address Windows code-signing enforcement.

AMD documents Smart App Control incompatibility in its
[Radeon Windows limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html).
The documented workaround is to turn Smart App Control off. This is a
**system-wide reduction in application execution protection**, requiring the
user's decision. The frontend never changes that setting.

If you choose that workaround, open **Windows Security > App & browser control
> Smart App Control settings**, select **Off**, and read the confirmation shown
by Windows before accepting it. Microsoft currently documents that recent
Windows updates support re-enabling the feature, while older Windows builds
can require a reset/reinstall. Check the warning and options on this machine;
do not assume the change is reversible on every build.

[Microsoft's Smart App Control FAQ](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions)
also says there are no exceptions for individual applications. A DLL with a
valid trusted publisher signature is the alternative to turning this feature
off; this investigation did not establish an available compatible signed
replacement for the installed AMD DLL.

After choosing a resolution, check the existing Python environment from the
repository root:

```powershell
.\.venv\Scripts\python.exe -B -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected results are the installed ROCm PyTorch version, `True`, and the AMD
GPU name. This checks runtime visibility without loading GeoRSCLIP or starting
embedding. Then reopen the desktop application and retry the query while the
embedding worker is idle/paused. Existing collection/GPU locking still applies.

No Windows security settings, Python packages, original imagery, embedding
bundles, or compatibility fingerprints were changed during this diagnosis.

After the progress-writer fix, the real `fighter jet` search completed with 10
matches across 1,220 published sources / 3,926,437 chips. The query used its
previously cached text vector, and both collection/GPU locks were confirmed
released afterward. All 19 frontend and 42 backend regressions passed; PyTorch
import was no longer blocked in the latest backend run.
