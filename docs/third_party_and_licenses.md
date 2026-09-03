# Third-Party Code and Licenses

## Method
Reviewed every source file for adapted/copied code, tutorial-derived
snippets, or non-standard third-party inclusions. Reviewed
`ml-service/requirements.txt`, `backend/package.json`, and
`frontend/index.html`'s CDN imports for library licenses.

## Findings

**No substantial third-party source code was identified beyond normal
library dependencies.** All application logic (data generation, feature
engineering, model training, the uplift/DR estimators, the policy
engine, the bandit, the FastAPI/Express services, the frontend) is
original to this project.

## Dependencies and their licenses

### Python (`ml-service/requirements.txt`)
| Package | License |
|---|---|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause |
| joblib | BSD-3-Clause |
| pydantic | MIT |
| requests | Apache-2.0 |

### Node (`backend/package.json`)
| Package | License |
|---|---|
| express | MIT |
| cors | MIT |
| axios | MIT |

### Frontend (`frontend/index.html`, CDN-loaded)
| Library | License |
|---|---|
| React / ReactDOM | MIT |
| Babel Standalone | MIT |

All dependencies use permissive licenses (MIT, BSD-3-Clause,
Apache-2.0) with no copyleft or attribution-in-UI requirements beyond
standard `LICENSE`/`NOTICE` file inclusion, which is satisfied by
depending on the published packages rather than vendoring their source.

## Conclusion

No restrictive-license or copied-code concerns. Standard, permissively
licensed dependencies only.
