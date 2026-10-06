"""Intervention plugins. Importing this package registers built-ins."""

from renewal.interventions.noop import NoopIntervention  # noqa: F401
from renewal.interventions.random_scaffolder import (  # noqa: F401
    RandomScaffolderIntervention,
)
from renewal.interventions.scaffolder import ScaffolderIntervention  # noqa: F401
