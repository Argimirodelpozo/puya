### Fixed

- Compiling no longer ends with "explicit condition check(s) removed during TEAL optimization" when an explicit check follows a call of a subroutine that never returns: the check cannot be reached, and removing it with the code after the call is not an error.
