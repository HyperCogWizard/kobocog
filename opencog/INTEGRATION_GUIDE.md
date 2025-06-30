# OpenCog-GGML Integration Guide

## Quick Start

This guide provides step-by-step instructions for integrating and using the OpenCog cognitive architecture with GGML tensors in Kobocog.

### 1. Build the OpenCog Components

```bash
cd /path/to/kobocog
make opencog-components
```

### 2. Build Kobocog with OpenCog Integration

```bash
# Build with OpenCog support
make koboldcpp_opencog

# Or add to existing build
make OPENCOG_ENABLED=1
```

### 3. Initialize the Cognitive System

```cpp
#include "opencog/api/cognitive_api.h"

// C++ Interface
opencog::KobocogIntegration::initialize_cognitive_api();
auto* api = opencog::KobocogIntegration::get_instance();

// C Interface  
opencog_init(256 * 1024 * 1024); // 256MB memory
```

### 4. Basic Usage Examples

#### Adding Knowledge

```cpp
// Add concepts
Handle cat = atomspace->add_concept("cat");
Handle animal = atomspace->add_concept("animal");

// Add inheritance relationship
Handle inheritance = atomspace->add_inheritance(cat, animal, 
    TruthValue(0.9f, 0.8f));
```

#### PLN Reasoning

```cpp
// Forward chaining
PLNResult result = pln_engine->forward_chain({premise1, premise2}, 10);

// Backward chaining  
PLNResult result = pln_engine->backward_chain(target_atom, 10);

// Apply specific rule
PLNResult result = pln_engine->apply_rule(PLNRuleType::DEDUCTION, 
    {premise1, premise2});
```

#### Attention Management

```cpp
// Focus attention on an atom
attention_system->focus_on_atom(important_atom, 1.0f);

// Get current attentional focus
auto focus_atoms = attention_system->get_focus();

// Spread activation
attention_system->spread_activation(source_atom, 0.1f);
```

### 5. REST API Usage

#### Memory Operations

```bash
# Add concept
curl -X POST http://localhost:5001/cognitive/memory \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "add_concept",
    "params": {"name": "robot"},
    "session_id": "my_session"
  }'

# Query concept
curl -X POST http://localhost:5001/cognitive/memory \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "query_concept", 
    "params": {"name": "robot"},
    "session_id": "my_session"
  }'
```

#### Reasoning Operations

```bash
# Forward chaining
curl -X POST http://localhost:5001/cognitive/reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "forward_chain",
    "params": {
      "premises": ["1001", "1002"],
      "max_steps": 5
    },
    "session_id": "my_session"
  }'
```

#### Attention Operations

```bash
# Get attentional focus
curl -X GET http://localhost:5001/cognitive/attention \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "get_focus",
    "session_id": "my_session"
  }'
```

## Architecture Overview

### Tensor Schema Design

The OpenCog-GGML integration maps cognitive structures to efficient tensor representations:

#### Node Types
- **ConceptNode**: 768-dimensional embedding vectors
- **PredicateNode**: 768-dimensional relational embeddings  
- **NumberNode**: Scalar values [1]
- **VariableNode**: 768-dimensional placeholder vectors

#### Link Types
- **InheritanceLink**: 768×768 relational matrices
- **EvaluationLink**: Variable-arity tensors [768, n]
- **ImplicationLink**: 768×768 logical relation matrices
- **SimilarityLink**: 768×768 symmetric similarity matrices

#### Attention Values
- **STI (Short-term Importance)**: Scalar [1]
- **LTI (Long-term Importance)**: Scalar [1] 
- **VLTI (Very Long-term Importance)**: Scalar [1]

### GGML Custom Operations

Eight specialized GGML operations support symbolic reasoning:

1. **`ggml_atomspace_lookup`**: Efficient atom retrieval by similarity
2. **`ggml_pattern_match`**: Structural pattern matching with thresholds
3. **`ggml_activation_spread`**: Hypergraph activation spreading
4. **`ggml_attention_update`**: ECAN attention value updates
5. **`ggml_pln_inference`**: PLN rule application in tensor space
6. **`ggml_hypergraph_conv`**: Convolution over hypergraph structures
7. **`ggml_symbolic_unify`**: Term unification for logical reasoning
8. **`ggml_truth_value_merge`**: PLN truth value combination

### PLN Reasoning Rules

The PLN engine supports all major probabilistic logic rules:

- **Deduction**: A→B, B→C ⊢ A→C
- **Induction**: A→B, A→C ⊢ B→C  
- **Abduction**: A→C, B→C ⊢ A→B
- **Revision**: Combine multiple estimates of same statement
- **Choice**: Select higher confidence estimate
- **Conjunction**: A∧B from A and B
- **Disjunction**: A∨B from A and B
- **Negation**: ¬A from A

### ECAN Attention System

The Economic Attention Network allocates cognitive resources using:

#### Attention Bank
- Economic stimulus allocation
- Rent collection for attention decay
- Attentional focus management
- Resource conservation

#### Attention Agents
- **Novelty Detection**: Identifies novel patterns
- **Importance Spreading**: Propagates attention through links
- **Forgetting**: Removes low-importance atoms
- **Reinforcement**: Rewards successful reasoning patterns

## Integration with Kobocog Core

### HTTP Server Integration

```cpp
// Register cognitive endpoints with Kobocog server
void register_cognitive_endpoints() {
    server.register_endpoint("/cognitive/memory", handle_memory_request);
    server.register_endpoint("/cognitive/reasoning", handle_reasoning_request);
    server.register_endpoint("/cognitive/attention", handle_attention_request);
    server.register_endpoint("/cognitive/status", handle_status_request);
}
```

### Koboldcpp.py Integration

```python
# Add cognitive parameters to Koboldcpp args
parser.add_argument("--opencog", action="store_true", 
                    help="Enable OpenCog cognitive reasoning")
parser.add_argument("--cognitive-memory", type=int, default=256,
                    help="Cognitive system memory in MB")
parser.add_argument("--attention-focus", type=int, default=100,
                    help="Attention focus size")
```

### Model Integration

The system can be integrated with existing language models:

```cpp
// Use cognitive system to enhance model reasoning
if (opencog_enabled) {
    // Query cognitive system for relevant knowledge
    auto cognitive_context = get_cognitive_context(input_text);
    
    // Integrate with model forward pass
    enhanced_input = combine_linguistic_and_cognitive(input_text, cognitive_context);
}
```

## Performance Considerations

### Memory Usage
- Base system: ~50MB for core components
- AtomSpace: ~200MB for 100K atoms with embeddings
- PLN cache: ~50MB for reasoning history
- Attention system: ~10MB for focus management

### Computational Complexity
- Atom lookup: O(log n) with tensor indexing
- PLN inference: O(k²) where k = premise count
- Attention update: O(n) where n = atom count
- Pattern matching: O(mn) where m,n = pattern sizes

### Optimization Tips

1. **Batch Operations**: Group multiple reasoning steps
2. **Attention Pruning**: Limit focus size to improve performance
3. **Lazy Evaluation**: Defer tensor computations until needed
4. **Memory Pooling**: Reuse GGML contexts across operations

## Advanced Features

### Custom Atom Types

```cpp
// Define custom atom type
enum class CustomAtomType {
    NEURAL_CONCEPT = 1000,
    TEMPORAL_LINK = 1001
};

// Implement custom tensor mapping
class NeuralConceptNode : public TensorAtom {
    static constexpr size_t EMBEDDING_DIM = 1024; // Larger embedding
    // ... implementation
};
```

### Custom PLN Rules

```cpp
// Define custom inference rule
PLNResult my_custom_rule(const std::vector<Handle>& premises) {
    // Custom rule logic
    TruthValue result_tv = compute_custom_truth_value(premises);
    return {conclusion_handle, result_tv, premises, result_tv.confidence, true};
}

// Register with PLN engine
pln_engine->register_custom_rule("my_rule", my_custom_rule);
```

### Attention Agents

```cpp
// Custom attention agent
class MyAttentionAgent : public AttentionAgent {
public:
    MyAttentionAgent() : AttentionAgent("MyAgent", 1.0f, 1) {}
    
    void run(TensorAtomSpace* atomspace, AttentionBank* bank) override {
        // Custom attention allocation logic
    }
};

// Add to attention system
attention_system->add_agent(std::make_unique<MyAttentionAgent>());
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase GGML context size
   ```cpp
   atomspace = new TensorAtomSpace(512 * 1024 * 1024); // 512MB
   ```

2. **Low Reasoning Performance**: Adjust PLN parameters
   ```cpp
   pln_engine->set_parameters(0.3f, 5); // Lower threshold, fewer steps
   ```

3. **Attention System Instability**: Tune ECAN parameters
   ```cpp
   attention_system->set_parameters(0.005f, 0.05f, 20); // Gentler decay
   ```

### Debug Mode

```cpp
#define OPENCOG_DEBUG 1
// Enables detailed logging of cognitive operations
```

### Performance Monitoring

```cpp
// Get system statistics
auto atomspace_stats = atomspace->get_stats();
auto pln_stats = pln_engine->get_statistics();
auto attention_stats = attention_system->get_statistics();

std::cout << "Atoms: " << atomspace_stats.num_atoms << std::endl;
std::cout << "Inferences: " << pln_stats.successful_inferences << std::endl;
std::cout << "Focus size: " << attention_stats.avg_focus_size << std::endl;
```

## Future Extensions

### Planned Features
- **MOSES Integration**: Evolutionary program synthesis
- **Temporal Reasoning**: Time-aware cognitive operations  
- **Multi-Modal Integration**: Vision and audio cognitive processing
- **Distributed Cognition**: Multi-agent cognitive systems
- **Neural-Symbolic Learning**: Gradient-based cognitive optimization

### Research Directions
- **Quantum Cognitive Computing**: Quantum-enhanced reasoning
- **Hierarchical Cognitive Architecture**: Multi-level cognitive processing
- **Consciousness Modeling**: Implementing models of machine consciousness
- **Cognitive Architectures**: Integration with other cognitive frameworks

## Contributing

### Development Setup
```bash
git clone https://github.com/HyperCogWizard/kobocog
cd kobocog
make opencog-components
make test-opencog
```

### Code Style
- Follow existing C++17 conventions
- Use RAII for memory management
- Document public APIs with Doxygen
- Add unit tests for new features

### Testing
```bash
# Run full test suite
make test-opencog

# Run specific test categories  
python3 opencog/tests/test_tensor_atomspace.py
python3 opencog/tests/test_pln_reasoning.py
python3 opencog/tests/test_attention_system.py
```

For questions and support, please open an issue or join our Discord community.