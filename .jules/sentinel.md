## 2025-03-09 - Avoid `shell=True` in self-healing engine
**Vulnerability:** The self-healing engine executed automatically generated shell commands (`self_healing.core.repair._execute_command` and `self_healing.core.validator._run_tests`) using `subprocess.run` with `shell=True`.
**Learning:** Using `shell=True` opens the system up to command injection (CWE-78) risks if the generated command or its parameters contain untrusted inputs.
**Prevention:** Avoid `shell=True` in `subprocess` calls by properly tokenizing shell command strings with `shlex.split()` and passing the list of arguments to `subprocess.run(..., shell=False)`.
