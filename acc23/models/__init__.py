"""
Model prototypes:

- Ampere (`acc23.models.ampere.Ampere`): Simple convolution & dense fusion
  model
- Gordon (`acc23.models.gordon.Gordon`): A fusion model at the convolution
  level
- London (`acc23.models.london.London`): Like Ampere but using a vision
  transformer
- Norway (`acc23.models.norway.Norway`): Like Ampere but using a co-attention
  vision transformer
- Orchid (`acc23.models.orchid.Orchid`): Uses a
  `acc23.models.transformers.TabTransformer` and a vision transformer
- Primus (`acc23.models.primus.Primus`): Like London but the categorical
  features are embedded

All models have sensible defaults, so you do not need to provide any argument
to instantiate them.
"""

from .ampere import Ampere
from .gordon import Gordon
from .london import London
from .norway import Norway
from .orchid import Orchid
from .primus import Primus
