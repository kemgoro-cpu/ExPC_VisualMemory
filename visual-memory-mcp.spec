from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("mcp")

a = Analysis(
    ["scripts/pyinstaller_mcp_entry.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=["mcp.server.fastmcp"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="visual-memory-mcp", console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="visual-memory-mcp")
