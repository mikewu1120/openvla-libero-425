"""Two Windows-only fixes for the pinned robosuite 1.4.1 dependency."""
import sys
import sysconfig
from pathlib import Path

if sys.platform == "win32":
    target = Path(sysconfig.get_paths()["purelib"]) / "robosuite/utils/binding_utils.py"
    source = target.read_text(encoding="utf-8")
    replacements = {
        'ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))':
        'ctypes.WinDLL(os.path.join(os.path.dirname(mujoco.__file__), "mujoco.dll"))',
        'if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:':
        'if _SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:',
    }
    for original, fixed in replacements.items():
        if original not in source and fixed not in source:
            raise RuntimeError("Unexpected robosuite version; expected 1.4.1")
        source = source.replace(original, fixed)
    target.write_text(source, encoding="utf-8")
    print(f"Windows DLL path and GLFW renderer fixes applied: {target}")

    # LIBERO's finally/continue hides non-randomization reset errors forever.
    wrapper = Path(__file__).resolve().parent / "LIBERO/libero/libero/envs/env_wrapper.py"
    source = wrapper.read_text(encoding="utf-8")
    source = source.replace('            except RandomizationError:\n                pass\n            finally:\n                continue\n',
                            '            except RandomizationError:\n                pass\n')
    wrapper.write_text(source, encoding="utf-8")
