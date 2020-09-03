.. _troubleshooting:

**************************
Troubleshooting
**************************

Import Errors
=============

ImportError: DLL load failed
----------------------------

Platform: Windows, error message:

    ImportError: DLL load failed: The specified module could not be found.

This usually happens when andes is not installed in a Conda environment
but instead in a system-wide Python whose library path was not correctly
set in environment variables.

The easiest fix is to install andes in a Conda environment.


Runtime Errors
==============

EOFError: Ran out of input
--------------------------

The error message looks like ::

    Traceback (most recent call last):
      File "/home/user/miniconda3/envs/andes/bin/andes", line 11, in <module>
        load_entry_point('andes', 'console_scripts', 'andes')()
      File "/home/user/repos/andes/andes/cli.py", line 179, in main
        return func(cli=True, **vars(args))
      File "/home/user/repos/andes/andes/main.py", line 514, in run
        system = run_case(cases[0], codegen=codegen, **kwargs)
      File "/home/user/repos/andes/andes/main.py", line 304, in run_case
        system = load(case, codegen=codegen, **kwargs)
      File "/home/user/repos/andes/andes/main.py", line 284, in load
        system.undill()
      File "/home/user/repos/andes/andes/system.py", line 980, in undill
        loaded_calls = self._load_pkl()
      File "/home/user/repos/andes/andes/system.py", line 963, in _load_pkl
        loaded_calls = dill.load(f)
      File "/home/user/miniconda3/envs/andes/lib/python3.7/site-packages/dill/_dill.py", line 270, in load
        return Unpickler(file, ignore=ignore, **kwds).load()
      File "/home/user/miniconda3/envs/andes/lib/python3.7/site-packages/dill/_dill.py", line 473, in load
        obj = StockUnpickler.load(self)
    EOFError: Ran out of input

Resolution:

The error indicates the file for generated code is corrupt or inaccessible.
It can be fixed by running ``andes prepare`` from the command line.

If the issue persists, try removing ``~/.andes/calls.pkl`` and running
``andes prepare`` agian.