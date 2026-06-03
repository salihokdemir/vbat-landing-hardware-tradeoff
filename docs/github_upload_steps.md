# GitHub upload steps

## 1. Create a new repository on GitHub

Suggested name:

```text
vbat-landing-hardware-tradeoff
```

Start as private if you want to review it before sharing.

## 2. Put this package in a local folder

Unzip the package and enter the folder:

```bash
cd vbat_github_release
```

## 3. Initialize git

```bash
git init
git add .
git commit -m "Initial final project reproducibility package"
```

## 4. Connect to GitHub

Replace `<username>` with your GitHub username:

```bash
git branch -M main
git remote add origin https://github.com/<username>/vbat-landing-hardware-tradeoff.git
git push -u origin main
```

## 5. Create a fixed final tag

```bash
git tag v1.0-final
git push origin v1.0-final
```

On GitHub, create a release from this tag. In the final report, cite the release or commit hash.

## 6. Optional archival DOI

After the GitHub release is ready, you can archive it through Zenodo to obtain a DOI. This is optional but improves reproducibility.
