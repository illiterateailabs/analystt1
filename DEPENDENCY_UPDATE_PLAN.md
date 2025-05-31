# Dependency Update Plan

_Last updated: **31 May 2025**_

---

## 1. Executive Summary

This document outlines a phased approach to updating the outdated Python dependencies in the `analyst-agent-illiterateai` project. Many packages are significantly behind their latest versions, posing potential risks related to security, performance, and compatibility, while also missing out on new features and bug fixes.

The update process will be conducted in stages to minimize disruption and allow for thorough testing at each step. The phases prioritize critical security updates first, followed by core framework components, data and ML libraries, development tools, and finally, other ecosystem packages.

**Key Goals:**
- Enhance security by patching known vulnerabilities.
- Improve performance and stability.
- Leverage new features and bug fixes from updated libraries.
- Maintain compatibility across the stack.
- Ensure a smooth transition with minimal breaking changes where possible.

---

## 2. Phase 1: Critical Security & Stability Updates (Immediate)

**Focus**: Address immediate security vulnerabilities and critical bug fixes. These updates are generally minor version bumps with high impact.

**Packages & Considerations**:

1.  **Pillow**: `10.1.0` → `10.4.0`
    *   **Considerations**: Includes security patches and new image format support. Upgrade strongly recommended.
2.  **aiohttp**: `3.9.1` → `3.10.5`
    *   **Considerations**: Performance optimizations and security fixes. Upgrade recommended.
3.  **sentry-sdk**: `1.38.0` → `2.16.0`
    *   **Considerations**: Major version jump. Enhanced error tracking and FastAPI integration. Review Sentry documentation for breaking changes in v2.x.
4.  **web3**: `6.12.0` → `7.3.0`
    *   **Considerations**: New Ethereum network support and bug fixes. Check for breaking changes in contract interaction if custom ABIs or older contract patterns are used.
5.  **jinja2**: `3.1.2` → `3.1.4`
    *   **Considerations**: Minor security patches and bug fixes. Upgrade recommended. Safe update.
6.  **python-multipart**: `0.0.6` → `0.0.12`
    *   **Considerations**: Newer versions improve form parsing and security. Upgrade recommended for FastAPI applications handling multipart data.
7.  **slowapi**: `0.1.8` → `0.1.9`
    *   **Considerations**: Minor updates with bug fixes and improved rate-limiting logic. Safe update.

**Testing Approach for Phase 1**:
*   Run all existing unit and integration tests, paying close attention to image processing, API request handling, error reporting, and blockchain interactions.
*   Manually test image upload and analysis features.
*   Verify Sentry error reporting.
*   Test multipart form data uploads if applicable.

---

## 3. Phase 2: Core Framework Updates (Timeline: 1-2 weeks)

**Focus**: Update the foundational web framework, data validation, and agent orchestration libraries. This phase is critical for overall application stability and performance.

**Packages & Considerations**:

1.  **fastapi**: `0.104.1` → `0.115.0`
    *   **Considerations**: Includes upgraded dependencies (e.g., Starlette 0.38.6), improved type annotations, and bug fixes. Python 3.6 support dropped. Generally safe but test routing and middleware.
2.  **uvicorn**: `0.24.0` → `0.32.0`
    *   **Considerations**: Performance optimizations and better async support. Standard extra includes uvloop. Should be compatible with new FastAPI.
3.  **pydantic**: `2.5.0` → `2.9.2`
    *   **Considerations**: Improved performance, better type checking, new features. Crucial for FastAPI. Review Pydantic v2 migration guide if any v1 features were still in use.
4.  **pydantic-settings**: `2.1.0` → `2.5.2`
    *   **Considerations**: Bug fixes and better integration with Pydantic 2.x. Update alongside Pydantic.
5.  **crewai**: `0.5.0` → `0.81.2`
    *   **Considerations**: Significant updates. **High risk of breaking changes**. Thoroughly review CrewAI changelogs for agent configuration, tool integration, and process execution. Test all crew workflows.
6.  **crewai-tools**: `>0.1.0` (current likely old) → `0.8.3`
    *   **Considerations**: Newer versions add tools. Update alongside `crewai`. Check if any custom tool implementations are affected.
7.  **e2b**: `0.15.0` → `1.0.3`
    *   **Considerations**: Major version jump (0.x to 1.x). **High risk of breaking changes**. Review e2b documentation for new sandboxing features, API changes, and potential impact on `SandboxExecutionTool`.
8.  **httpx**: `0.25.2` → `0.27.2`
    *   **Considerations**: Improved async support and bug fixes. Check compatibility with FastAPI and other clients.
9.  **sqlalchemy**: `2.0.23` → `2.0.36`
    *   **Considerations**: Enhanced async support and bug fixes. Generally safe within 2.0.x.
10. **alembic**: `1.13.1` → `1.13.3`
    *   **Considerations**: Minor updates for migration stability. Safe update.
11. **asyncpg**: `0.29.0` → `0.30.0`
    *   **Considerations**: Performance improvements for PostgreSQL async queries. Safe update.
12. **redis**: `5.0.1` → `5.1.1`
    *   **Considerations**: Includes bug fixes and performance improvements. Check Redis client interactions.

**Testing Approach for Phase 2**:
*   Run all existing tests. Focus on API endpoint functionality, request/response validation, agent execution, crew workflows, and e2b sandbox interactions.
*   Manually test all major API endpoints and agent-driven processes.
*   Verify database interactions and migrations.
*   Monitor performance for core operations.

---

## 4. Phase 3: ML/Data Libraries (Timeline: 2-3 weeks)

**Focus**: Update libraries central to data manipulation, machine learning, and graph analytics. This phase carries a risk of subtle changes in algorithm behavior or API incompatibilities.

**Packages & Considerations**:

1.  **numpy**: `1.25.2` → `2.1.1`
    *   **Considerations**: **Major version jump (1.x to 2.x)**. High risk of breaking changes, especially C API updates. Ensure compatibility with Pandas, SciPy, Scikit-learn, PyTorch, etc. Test numerical stability.
2.  **pandas**: `2.1.4` → `2.2.3`
    *   **Considerations**: Performance optimizations and new features. Update after or alongside NumPy.
3.  **scipy**: `1.11.4` → `1.14.1`
    *   **Considerations**: New algorithms and optimizations. Check compatibility with new NumPy.
4.  **scikit-learn**: `1.3.2` → `1.5.2`
    *   **Considerations**: Adds new ML algorithms and improved performance. Test existing model training and prediction pipelines.
5.  **networkx**: `3.2.1` → `3.4.1`
    *   **Considerations**: New graph algorithms and performance improvements. Check graph analysis functions.
6.  **torch**: `2.1.1` → `2.5.0`
    *   **Considerations**: Major updates. Check compatibility with `torch-geometric` and `dgl`. Test model training and inference.
7.  **torch-geometric**: `2.4.0` → `2.6.1`
    *   **Considerations**: Improved GNN support. Update after `torch`.
8.  **dgl**: `1.1.3` → `2.4.0`
    *   **Considerations**: Major updates with enhanced graph ML features. Update after `torch`.
9.  **statsmodels**: `0.14.0` → `0.14.3`
    *   **Considerations**: Bug fixes and new statistical models. Safe update.
10. **chromadb**: `>0.5.23` (current likely old) → `0.7.0`
    *   **Considerations**: Improves vector search and indexing. Test ChromaDB interactions and query performance.

**Testing Approach for Phase 3**:
*   Run all tests, focusing on data processing pipelines, ML model training (`FraudMLTool`), prediction accuracy, graph algorithms (`CryptoAnomalyTool`, `GraphQueryTool`), and any numerical computations.
*   Retrain and evaluate key ML models to check for performance changes.
*   Validate outputs of graph analysis and data manipulation steps.

---

## 5. Phase 4: Development Tools & Utilities (Timeline: 3-4 weeks)

**Focus**: Update development tools, linters, formatters, and various utility libraries. Generally lower risk but can introduce new linting rules or minor behavioral changes.

**Packages & Considerations**:

1.  **black**: `23.11.0` → `24.8.0`
    *   **Considerations**: New formatting rules. Run `black .` after update.
2.  **isort**: `5.12.0` → `5.13.2`
    *   **Considerations**: Bug fixes. Run `isort .` after update.
3.  **ruff**: `0.1.5` → `0.7.0`
    *   **Considerations**: Significant performance improvements and new linting rules. May require code adjustments to satisfy new rules.
4.  **mypy**: `1.7.1` → `1.12.0`
    *   **Considerations**: Enhanced type checking and performance. May reveal new type errors.
5.  **pre-commit**: `3.5.0` → `4.0.1`
    *   **Considerations**: Improved hook management. Update hooks if necessary.
6.  **pytest**: `7.4.3` → `8.3.3`
    *   **Considerations**: Improved async testing and plugins. Check plugin compatibility.
7.  **pytest-asyncio**: `0.21.1` → `0.24.0`
    *   **Considerations**: Better async test support. Update with Pytest.
8.  **pytest-cov**: `4.1.0` → `5.0.0`
    *   **Considerations**: Enhanced coverage reporting.
9.  **pytest-env**: `1.1.1` → `1.1.5`
    *   **Considerations**: Minor updates. Safe update.
10. **structlog**: `23.2.0` → `24.4.0`
    *   **Considerations**: Improved formatting and performance. Check log output.
11. **prometheus-client**: `0.17.1` → `0.21.0`
    *   **Considerations**: New metrics and performance improvements. Check custom metric registration.
12. **python-dotenv**: `1.0.0` → `1.0.1`
    *   **Considerations**: Minor bug fixes. Safe update.
13. **python-dateutil**: `2.8.2` → `2.9.0`
    *   **Considerations**: Bug fixes and new date parsing features.
14. **pytz**: `2023.3` → `2024.2`
    *   **Considerations**: Updated timezone data.
15. **rich**: `13.7.0` → `13.9.2`
    *   **Considerations**: New formatting options and bug fixes.
16. **tqdm**: `4.66.1` → `4.66.5`
    *   **Considerations**: Minor UI improvements.
17. **aiofiles**: `23.2.1` → `24.1.0`
    *   **Considerations**: Improved async file handling.
18. **opencv-python**: `4.8.1.78` → `4.10.0.84`
    *   **Considerations**: Performance improvements and new CV algorithms. Test image processing.
19. **pytesseract**: `0.3.10` → `0.3.13`
    *   **Considerations**: Bug fixes and improved OCR accuracy. Test OCR functionality.

**Testing Approach for Phase 4**:
*   Run CI pipeline (linting, type-checking, tests).
*   Address any new linting or type errors.
*   Verify logging output and Prometheus metrics.
*   Check CLI tool behavior if `click` or `rich` changes affect them.

---

## 6. Phase 5: Remaining Ecosystem & Visualization Updates (Optional / Lower Priority)

**Focus**: Update remaining packages, including those with significant version jumps or where updates are less critical for core functionality.

**Packages & Considerations**:

1.  **google-cloud-aiplatform**: `1.38.1` → `1.69.0`
    *   **Considerations**: Significant updates with new Vertex AI features. **High risk of breaking changes**. Carefully review changelogs and test any direct Vertex AI API calls.
2.  **neo4j (driver)**: `5.15.0` → `5.24.0`
    *   **Considerations**: Performance improvements and new driver features. Generally safe within major version.
3.  **matplotlib**: `3.8.2` → `3.9.2`
    *   **Considerations**: Minor updates. Check plot outputs.
4.  **plotly**: `5.17.0` → `5.24.1`
    *   **Considerations**: New chart types. Check plot outputs.
5.  **seaborn**: `0.13.0` → `0.13.2`
    *   **Considerations**: Minor bug fixes. Check plot outputs.
6.  **eth-account**: `0.9.0` → `0.14.0`
    *   **Considerations**: Improved account management and security. Test any direct crypto account operations.
7.  **yfinance**: `0.2.28` → `0.2.44`
    *   **Considerations**: Bug fixes and new data endpoints. Test financial data fetching.
8.  **alpha-vantage**: `2.3.1` → `3.0.0`
    *   **Considerations**: Major update. Check for breaking changes in API interaction.
9.  **spacy**: `3.6.1` → `3.8.0`
    *   **Considerations**: New models and performance improvements. Check model compatibility and NLP pipeline outputs.
10. **transformers**: `4.35.2` → `4.45.1`
    *   **Considerations**: New models and optimizations. Update after/with spaCy if there are interdependencies.
11. **websockets**: `12.0` → `13.1`
    *   **Considerations**: Improved WebSocket protocol handling. Test any WebSocket communications.

**Testing Approach for Phase 5**:
*   Run all tests.
*   Focus on specific integrations: Vertex AI, Neo4j driver features, visualization outputs, financial data APIs, NLP pipelines.

---

## 7. Risk Mitigation Strategies

*   **Incremental Updates**: Update one package or a small, related group of packages at a time.
*   **Dedicated Branches**: Perform each update in a separate Git branch.
*   **Thorough Testing**: Run all automated tests (unit, integration) after each update. Perform manual testing of affected areas.
*   **Changelog Review**: Carefully read the changelogs for each package, especially for breaking changes.
*   **Virtual Environments**: Use fresh virtual environments to test updates in isolation.
*   **Pinning Transitive Dependencies**: If resolution issues arise, explicitly pin problematic transitive dependencies in `constraints.txt`.
*   **Rollback Plan**: Be prepared to revert an update if critical issues are found that cannot be quickly resolved.
*   **CI/CD Monitoring**: Closely monitor CI/CD pipelines after merging updates.

---

## 8. General Testing Approach for Each Phase

1.  **Pre-Update Benchmark**: Ensure all existing tests pass on the current branch before starting an update. Note current performance metrics if applicable.
2.  **Update Package(s)**: Modify `requirements.txt` and/or `constraints.txt`.
3.  **Rebuild Environment**: Create a fresh virtual environment and install updated dependencies (`pip install -r requirements.txt -c constraints.txt`).
4.  **Run Linters & Type Checkers**: Execute `ruff`, `black`, `isort`, `mypy`. Address any new issues.
5.  **Run Automated Tests**: Execute the full `pytest` suite. Debug any failures.
6.  **Manual Smoke Testing**: Perform manual testing of core application workflows, especially those likely affected by the updated package(s).
    *   Authentication flow (login, register, protected routes).
    *   Core agent execution paths.
    *   Data ingestion and processing.
    *   Key API endpoints.
7.  **Review Outputs**: Check logs, generated reports, database entries, and UI elements for correctness.
8.  **Performance Check**: (Optional, for major updates) Compare performance against pre-update benchmarks.
9.  **Merge & Monitor**: If all tests pass and manual verification is successful, merge the update branch. Monitor production/staging environments closely after deployment.

---

This plan provides a structured way to bring the project's dependencies up to date, balancing the benefits of new versions with the risks of introducing instability. Each phase should be approached methodically.
