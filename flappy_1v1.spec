# flappy_1v1.spec
# Run with:  pyinstaller flappy_1v1.spec

import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE

block_cipher = None

a = Analysis(
    ['flappy_1v1.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('evo_checkpoint.npz', '.'),
        ('Flappy_env.py', '.'),
        ('Flappy_evo.py', '.'),
    ],
    hiddenimports=['numpy', 'pygame', 'socket', 'threading', 'json'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Flappy1v1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
