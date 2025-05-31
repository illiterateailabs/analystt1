# GitLab CI Dependency Resolution and Timeout Fixes

_Date: **31 May 2025**_

This document outlines the issues encountered with the GitLab CI pipeline, specifically `lint` job timeouts, and the corrective actions taken.

---

## 1. Problem Statement

The `lint` job in the GitLab CI pipeline was consistently timing out after 1 hour. The primary cause was identified as the `pip install -r requirements.txt --constraint constraints.txt` command taking an excessively long time to resolve dependencies.

**Log Snippet Indicating Backtracking:**
```
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
```

---

## 2. Root Cause Analysis

The extended dependency resolution time was due to `pip`'s backtracking mechanism. With a complex set of direct and transitive dependencies (particularly from packages like `google-cloud-aiplatform`, `langchain`, `crewai`, and `transformers`), `pip` was attempting to evaluate hundreds of version combinations for several packages to find a compatible set.

Packages observed to cause significant backtracking included:
- `grpcio-status`
- `langchain-core`
- `langchain-community`
- `huggingface-hub`
- `googleapis-common-protos`
- `protobuf`

Additionally, a separate configuration error was found in the `test` job's `services` definition where `ports` were specified for the `neo4j` service, which is not a valid configuration key in GitLab CI services.

---

## 3. Solutions Implemented

### 3.1. Enhanced Dependency Constraints

To mitigate `pip` backtracking, explicit version constraints were added to `constraints.txt` for the problematic packages and their related dependencies. This significantly reduces the search space for the resolver.

**Key additions to `constraints.txt`:**
```
# ... (other constraints) ...

# Constraints added to mitigate pip backtracking issues observed in CI (May 31, 2025)
# These help stabilize the dependency resolution for complex packages like
# google-cloud-aiplatform, langchain, crewai, and transformers.
grpcio~=1.62.2
grpcio-status~=1.62.2
langchain-core~=0.1.52
langchain-community~=0.0.38
huggingface-hub~=0.20.3
googleapis-common-protos~=1.69.0
# Pinning protobuf as it's a common source of conflict with google libraries
protobuf~=4.25.0
```
*Commit SHA for this change (and initial CI optimizations): `1786dbfa08b34d403d4564266185589b9602b309`*

### 3.2. GitLab CI Configuration (`.gitlab-ci.yml`) Optimizations

Several changes were made to the `.gitlab-ci.yml` file:

1.  **Removed Disallowed `ports` Key**: The `ports` mapping was removed from the `neo4j` service definition in the `test` job, as this is not supported by GitLab CI services.
    *Commit SHA for this fix: `487f7eace904e59fda794d1384394c1944c37f62`*

2.  **Lint Job Timeout**: A `timeout: 30m` was added to the `lint` job as a safeguard, though the dependency pinning should prevent this from being reached.

3.  **Optimized Test Dependency Installation**: Dependencies for the `test` job (which runs in a matrix for Python 3.9, 3.10, 3.11) are now installed once per Python version using a `before_script` block within the `test` job definition. This avoids redundant installations for each parallel job instance if the cache is effective.

4.  **Improved Cache Key**: The cache key was updated to include `requirements.txt` and `constraints.txt` to ensure the cache is more accurately invalidated when dependencies change:
    ```yaml
    cache:
      key:
        files:
          - requirements.txt
          - constraints.txt
        prefix: files-$CI_COMMIT_REF_SLUG-$PYTHON_VERSION
      paths:
        - .cache/pip/
      policy: pull-push
    ```
    *These CI optimizations were part of commit SHA: `1786dbfa08b34d403d4564266185589b9602b309` and refined in `487f7eace904e59fda794d1384394c1944c37f62`.*

---

## 4. Manual Steps for GitLab Synchronization

Since these changes were committed to the GitHub repository (`illiterateailabs/analyst-agent-illiterateai`), they need to be manually synced to your GitLab repository (`illiterateailabs/analyst-agent-illiterateai-gitlabs`).

Execute the following commands in your local clone of the repository:
```bash
# Ensure your local main branch is up-to-date with GitHub
git checkout main
git pull origin main

# Push the changes to your GitLab remote
# (Assuming 'gitlab' is the name of your GitLab remote)
git push gitlab main
```

---

## 5. Expected Outcome

After applying these changes and syncing to GitLab, the CI pipeline, particularly the `lint` and `test` stages, should:
- Complete significantly faster due to reduced dependency resolution time.
- Be more reliable and less prone to timeouts.
- Utilize caching more effectively.

If timeouts persist, further investigation into the specific environment or runner constraints on GitLab might be necessary. However, these changes address the most common causes of `pip`-related performance issues in CI.
