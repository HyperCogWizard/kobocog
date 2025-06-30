# OpenCog-GGML Integration for Kobocog Core

This directory contains the implementation of OpenCog cognitive architecture components integrated with GGML tensor operations for distributed neural-symbolic reasoning.

## Architecture Overview

### Core Components

1. **AtomSpace Tensor Mapping** (`atomspace/`)
   - Maps OpenCog AtomSpace nodes and links to GGML tensor representations
   - Implements hypergraph tensor encoding for cognitive state representation

2. **GGML Custom Operations** (`operations/`)
   - Custom GGML kernels for AtomSpace manipulation
   - Pattern matching and activation spreading operations
   - Neural-symbolic bridge operations

3. **Reasoning Engines** (`reasoning/`)
   - PLN (Probabilistic Logic Networks) tensorized inference
   - MOSES evolutionary program synthesis
   - Neural reasoning with attention mechanisms

4. **Attention System** (`attention/`)
   - ECAN (Economic Attention Allocation) implementation
   - Real-time attention updates in GGML computation graphs
   - Dynamic focus on salient nodes/subgraphs

5. **API Integration** (`api/`)
   - Kobocog API endpoints for cognitive operations
   - Scheme wrapper interface for OpenCog logic
   - Memory and reasoning operation endpoints

## Tensor Schema Design

### AtomSpace Node Mapping
- **ConceptNode**: Fixed-size embeddings [768]
- **PredicateNode**: Relational embeddings [768]
- **NumberNode**: Scalar tensors [1]
- **VariableNode**: Placeholder tensors [768]

### AtomSpace Link Mapping
- **InheritanceLink**: Relation tensor [768, 768]
- **EvaluationLink**: Evaluation tensor [768, arity]
- **ImplicationLink**: Logical implication [768, 768]
- **SimilarityLink**: Similarity matrix [768, 768]

### Attention Values
- **STI**: Short-term importance tensor [1]
- **LTI**: Long-term importance tensor [1]
- **VLTI**: Very long-term importance tensor [1]

## Usage

The OpenCog-GGML integration provides both C++ and Python interfaces:

```cpp
// C++ Interface
#include "opencog/atomspace/tensor_atomspace.h"
#include "opencog/reasoning/pln_tensor.h"

// Create tensor-backed AtomSpace
TensorAtomSpace* tas = new TensorAtomSpace(ctx);

// Add concept with tensor representation
Handle concept = tas->add_concept("cat", embedding_tensor);

// Perform PLN inference
PLNTensorEngine pln(tas);
Handle result = pln.infer(premise1, premise2);
```

```python
# Python Interface
from opencog.kogni import TensorAtomSpace, PLNEngine

# Initialize cognitive system
atomspace = TensorAtomSpace()
pln = PLNEngine(atomspace)

# Add knowledge
cat = atomspace.add_concept("cat")
animal = atomspace.add_concept("animal")
atomspace.add_inheritance(cat, animal, confidence=0.9)

# Perform reasoning
result = pln.backward_chain(query)
```

## Integration with Kobocog

The OpenCog integration extends Kobocog's existing API with cognitive reasoning endpoints:

- `POST /cognitive/reasoning` - Perform symbolic reasoning
- `POST /cognitive/memory` - AtomSpace memory operations
- `GET /cognitive/attention` - Query attention allocation
- `POST /cognitive/learn` - Adaptive learning operations

## Testing

Run the test suite with:
```bash
cd opencog/tests
python test_tensor_atomspace.py
python test_pln_integration.py
python test_attention_system.py
```

## Dependencies

- GGML (included in Kobocog)
- OpenCog (optional, for Scheme interface)
- Python 3.8+ (for Python bindings)