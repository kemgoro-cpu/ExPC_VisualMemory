from __future__ import annotations

import os
from subprocess import CompletedProcess

import pytest

from visual_memory.service import bitlocker_status


@pytest.mark.skipif(os.name != "nt", reason="BitLocker is Windows-only")
def test_bitlocker_drive_is_passed_as_a_powershell_argument(monkeypatch, tmp_path):
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return CompletedProcess(
            command,
            0,
            stdout='{"ProtectionStatus":"On","VolumeStatus":"FullyEncrypted"}',
            stderr="",
        )

    monkeypatch.setattr("visual_memory.service.subprocess.run", fake_run)

    result = bitlocker_status(tmp_path)

    assert result["status"] == "protected"
    assert "$args[0]" in captured[4]
    assert captured[5] == tmp_path.resolve().drive
    assert captured[5] not in captured[4]
