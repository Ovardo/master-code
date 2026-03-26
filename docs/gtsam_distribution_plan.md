# GTSAM Distribution Plan

This project currently depends on a custom GTSAM build from a feature branch with extra Python wrapper support. That is manageable during development, but it is a poor fit for a student-facing install flow across macOS, Linux, and Windows.

The goal is to keep development flexible now, while making the final assignment install feel like a normal Python package install later.

## Recommended Release Shape

- Keep the custom GTSAM wheel build separate from this thesis package.
- Build and publish binary wheels for your patched GTSAM fork close to the assignment release date.
- Make this project depend on that custom wheel package once the API and wrappers have stabilized.

Recommended package name:

- `gtsam-ttk4250`

Avoid publishing the patched build under plain `gtsam`, since that makes it too easy to confuse with the public upstream `4.2` wheels.

## Support Matrix

Keep the support matrix intentionally small for the course:

- Python: `3.11` only
- macOS: Apple Silicon and Intel
- Linux: `x86_64`
- Windows: prefer WSL2 unless native wheels are validated and stable

Using one Python version removes a lot of wheel and troubleshooting complexity. If students need Windows, WSL2 is usually a much lower-risk target than native Windows for scientific Python plus C++ bindings.

## Near-Term Setup

Until the custom wheel exists:

- Continue using your current manual GTSAM fork build for development.
- Keep this repository installable without bundling GTSAM itself.
- Use `check-gtsam-install` to verify that the expected wrapper symbols are available.

Example:

```bash
python -m pip install -e .
check-gtsam-install --expect gtsam.ISAM2.jointMarginalCovariance
```

That check is aimed at the patched `ISAM2.jointMarginalCovariance(...)` wrapper exposed by your fork.

## Release-Time Setup

When the GTSAM branch has settled:

1. Build wheels from the patched GTSAM fork.
2. Publish those wheels to GitHub Releases or a package index.
3. Update this repository to depend on `gtsam-ttk4250` instead of upstream `gtsam`.
4. Freeze the assignment support matrix in the handout.

Student install should then look like one of these:

```bash
python -m pip install master-thesis-slam --find-links <release-url>
```

or, if the thesis package itself is not published:

```bash
python -m pip install --find-links <release-url> -e .
```

## Repository Split

Use the repositories like this:

- GTSAM fork: source of truth for the C++ change and Python wrapper
- This repo: assignment code, docs, smoke tests, and student instructions

That split keeps the heavy C++ wheel build separate from the assignment repository, and it lets you rebuild wheels later without changing the coursework code.

## CI Framework

The template workflow lives at:

- `packaging/templates/gtsam_custom_wheels.yml`

It is intentionally not active yet. When you are ready:

1. Move it into `.github/workflows/`
2. Point it at your GTSAM fork and branch
3. Fill in the actual build and install commands for your fork
4. Add release upload steps

The template uses `cibuildwheel`, because that is the standard way to automate Python wheel builds for macOS, Linux, and Windows.

## Release Checklist

- Pick one supported Python version
- Confirm the custom wrapper API is stable
- Decide whether Windows means native wheels or WSL2 only
- Build test wheels for macOS and Linux first
- Smoke-test wheel installs in fresh virtual environments
- Update this repo dependency from upstream `gtsam` to `gtsam-ttk4250`
- Replace the manual GTSAM build instructions in the student handout

## Student-Facing Guidance

For the assignment handout, keep setup rules simple:

- use Python `3.11`
- use a fresh virtual environment
- install with the provided command only
- run `check-gtsam-install` before running the SLAM scripts
- if on Windows, use WSL2 unless native Windows is explicitly supported

That combination will save you a lot of support time.
