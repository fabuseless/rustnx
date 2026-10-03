# Security policy

## Supported versions

rustnx is in alpha. Security fixes go into the latest release only.

| Version | Supported |
|---|---|
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Please don't report security problems in public issues.

Report them privately through GitHub instead: on the
[Security tab](https://github.com/fabuseless/rustnx/security) of this
repository, click **Report a vulnerability**. Only the maintainers can see the
report.

Please include:

- what the problem is and what an attacker could do with it;
- the rustnx, NetworkX and Python versions, and your operating system;
- steps or a small script that reproduce it.

## What to expect

- An acknowledgement within 7 days.
- An assessment, and a plan if a fix is needed, within 30 days.
- Credit in the release notes when the fix is published, unless you'd rather
  stay anonymous.

## Scope

rustnx runs graph algorithms on data you pass it. Things worth reporting
include memory-safety problems in the Rust extension, crashes or hangs caused
by crafted input (for example, crafted data passed to `pickle.loads` for a
rustnx graph), and problems in how release packages are built or published.
Wrong algorithm results that aren't security issues are bugs: please open a
normal issue for those.
