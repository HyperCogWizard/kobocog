#pragma once

/**
 * @file moses_tensor.h
 * @brief MOSES (Meta-Optimizing Semantic Evolutionary Search) using GGML tensors
 * 
 * This file implements evolutionary program synthesis for learning and optimization
 * in the OpenCog cognitive architecture using efficient tensor operations.
 */

#include "../atomspace/tensor_atomspace.h"
#include "../operations/ggml_opencog_ops.h"
#include <vector>
#include <memory>
#include <functional>
#include <random>

namespace opencog {

/**
 * @brief Program representation for evolutionary search
 */
struct TensorProgram {
    Handle root_atom;                    // Root atom of the program
    std::vector<Handle> subprograms;     // Component atoms
    struct ggml_tensor* representation; // Tensor encoding of program
    float fitness;                       // Fitness score
    size_t complexity;                   // Program complexity measure
    
    TensorProgram(Handle root = INVALID_HANDLE) 
        : root_atom(root), representation(nullptr), fitness(0.0f), complexity(0) {}
};

/**
 * @brief Population of programs for evolutionary search
 */
class ProgramPopulation {
private:
    std::vector<std::unique_ptr<TensorProgram>> programs_;
    size_t max_size_;
    struct ggml_context* ctx_;
    
public:
    explicit ProgramPopulation(struct ggml_context* ctx, size_t max_size = 1000);
    ~ProgramPopulation();
    
    void add_program(std::unique_ptr<TensorProgram> program);
    void remove_program(size_t index);
    TensorProgram* get_program(size_t index) const;
    size_t size() const { return programs_.size(); }
    
    // Population operations
    void sort_by_fitness();
    std::vector<TensorProgram*> get_elite(size_t count) const;
    std::vector<TensorProgram*> tournament_selection(size_t tournament_size, size_t count);
    void clear_weak_programs(float fitness_threshold);
    
    // Statistics
    float get_avg_fitness() const;
    float get_best_fitness() const;
    size_t get_avg_complexity() const;
};

/**
 * @brief Fitness evaluation function type
 */
using FitnessFunction = std::function<float(const TensorProgram&, TensorAtomSpace*)>;

/**
 * @brief MOSES evolutionary search parameters
 */
struct MOSESParams {
    size_t population_size = 500;
    size_t max_generations = 100;
    float mutation_rate = 0.1f;
    float crossover_rate = 0.7f;
    float elite_fraction = 0.1f;
    size_t tournament_size = 5;
    size_t max_program_size = 50;
    float complexity_penalty = 0.01f;
    bool adaptive_rates = true;
    
    MOSESParams() = default;
};

/**
 * @brief Main MOSES evolutionary program synthesis engine
 */
class MOSESTensorEngine {
private:
    TensorAtomSpace* atomspace_;
    struct ggml_context* ctx_;
    std::unique_ptr<ProgramPopulation> population_;
    MOSESParams params_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
    std::normal_distribution<float> normal_dist_;
    
    // Evolution statistics
    struct MOSESStats {
        size_t generation_count;
        float best_fitness_ever;
        float avg_fitness_history[100]; // Last 100 generations
        size_t evaluations_count;
        size_t successful_crossovers;
        size_t successful_mutations;
    };
    
    mutable MOSESStats stats_;

public:
    /**
     * @brief Constructor
     */
    explicit MOSESTensorEngine(TensorAtomSpace* atomspace);
    
    /**
     * @brief Destructor
     */
    ~MOSESTensorEngine();
    
    /**
     * @brief Set evolution parameters
     */
    void set_parameters(const MOSESParams& params);
    
    /**
     * @brief Initialize population with random programs
     */
    void initialize_population(const std::vector<AtomType>& allowed_types,
                              size_t max_depth = 5);
    
    /**
     * @brief Run evolutionary search
     */
    TensorProgram* evolve(FitnessFunction fitness_func, 
                         size_t max_generations = 0);
    
    /**
     * @brief Evaluate single program
     */
    float evaluate_program(TensorProgram& program, FitnessFunction fitness_func);
    
    /**
     * @brief Create offspring through crossover
     */
    std::unique_ptr<TensorProgram> crossover(const TensorProgram& parent1,
                                           const TensorProgram& parent2);
    
    /**
     * @brief Mutate program
     */
    void mutate(TensorProgram& program, float mutation_strength = 1.0f);
    
    /**
     * @brief Add program to population
     */
    void add_program(std::unique_ptr<TensorProgram> program);
    
    /**
     * @brief Get best program found so far
     */
    TensorProgram* get_best_program() const;
    
    /**
     * @brief Get evolution statistics
     */
    MOSESStats get_statistics() const;
    
    /**
     * @brief Reset evolution state
     */
    void reset();

private:
    // Program generation
    std::unique_ptr<TensorProgram> generate_random_program(
        const std::vector<AtomType>& allowed_types, size_t max_depth);
    Handle create_random_atom(const std::vector<AtomType>& allowed_types);
    void build_program_tree(TensorProgram& program, Handle root, size_t depth, size_t max_depth);
    
    // Genetic operations
    Handle crossover_subtrees(Handle tree1, Handle tree2, float crossover_point);
    void mutate_subtree(Handle& root, float mutation_strength);
    Handle random_subtree_replacement(Handle original, const std::vector<AtomType>& types);
    
    // Tensor operations
    struct ggml_tensor* encode_program(const TensorProgram& program);
    void decode_program(TensorProgram& program, struct ggml_tensor* encoding);
    
    // Utility functions
    size_t calculate_complexity(const TensorProgram& program);
    float complexity_penalty(size_t complexity);
    void update_statistics(float fitness);
    
    // Selection methods
    std::vector<TensorProgram*> roulette_wheel_selection(size_t count);
    std::vector<TensorProgram*> rank_selection(size_t count);
    
    // Adaptive parameter adjustment
    void adapt_mutation_rate(float fitness_improvement);
    void adapt_crossover_rate(float population_diversity);
};

/**
 * @brief Predefined fitness functions for common tasks
 */
class MOSESFitnessFunctions {
public:
    /**
     * @brief Boolean satisfiability fitness
     */
    static float boolean_sat_fitness(const TensorProgram& program, TensorAtomSpace* atomspace);
    
    /**
     * @brief Symbolic regression fitness
     */
    static float symbolic_regression_fitness(const TensorProgram& program, 
                                           TensorAtomSpace* atomspace,
                                           const std::vector<std::pair<float, float>>& data);
    
    /**
     * @brief Classification fitness
     */
    static float classification_fitness(const TensorProgram& program,
                                      TensorAtomSpace* atomspace,
                                      const std::vector<std::pair<std::vector<float>, int>>& data);
    
    /**
     * @brief Theorem proving fitness
     */
    static float theorem_proving_fitness(const TensorProgram& program,
                                        TensorAtomSpace* atomspace,
                                        Handle target_theorem);
    
    /**
     * @brief Program synthesis fitness (for learning algorithms)
     */
    static float program_synthesis_fitness(const TensorProgram& program,
                                         TensorAtomSpace* atomspace,
                                         const std::vector<std::pair<std::vector<float>, std::vector<float>>>& io_examples);
    
    /**
     * @brief Knowledge completion fitness
     */
    static float knowledge_completion_fitness(const TensorProgram& program,
                                            TensorAtomSpace* atomspace,
                                            const std::vector<Handle>& missing_knowledge);
};

/**
 * @brief Specialized MOSES applications
 */

/**
 * @brief Learn PLN rules from data
 */
class PLNRuleLearner {
private:
    MOSESTensorEngine* moses_engine_;
    std::vector<std::pair<std::vector<Handle>, Handle>> training_data_;
    
public:
    explicit PLNRuleLearner(MOSESTensorEngine* engine);
    
    void add_training_example(const std::vector<Handle>& premises, Handle conclusion);
    TensorProgram* learn_rule(size_t max_generations = 50);
    float evaluate_rule_fitness(const TensorProgram& rule);
};

/**
 * @brief Learn attention allocation strategies
 */
class AttentionStrategyLearner {
private:
    MOSESTensorEngine* moses_engine_;
    std::vector<std::pair<std::vector<Handle>, std::vector<float>>> attention_data_;
    
public:
    explicit AttentionStrategyLearner(MOSESTensorEngine* engine);
    
    void add_attention_example(const std::vector<Handle>& atoms, 
                              const std::vector<float>& optimal_attention);
    TensorProgram* learn_strategy(size_t max_generations = 50);
    float evaluate_strategy_fitness(const TensorProgram& strategy);
};

/**
 * @brief Meta-learning for cognitive architecture optimization
 */
class CognitiveMetaLearner {
private:
    MOSESTensorEngine* moses_engine_;
    TensorAtomSpace* atomspace_;
    
    struct CognitiveTask {
        std::string name;
        FitnessFunction fitness_func;
        std::vector<Handle> relevant_atoms;
        float difficulty;
    };
    
    std::vector<CognitiveTask> tasks_;
    
public:
    explicit CognitiveMetaLearner(MOSESTensorEngine* engine, TensorAtomSpace* atomspace);
    
    void add_task(const std::string& name, FitnessFunction fitness_func, 
                  const std::vector<Handle>& atoms, float difficulty = 1.0f);
    
    TensorProgram* learn_meta_strategy(size_t max_generations = 100);
    float evaluate_meta_fitness(const TensorProgram& meta_program);
    
    // Task transfer learning
    void transfer_knowledge(const TensorProgram& source_program, 
                           const CognitiveTask& target_task);
};

/**
 * @brief MOSES integration with PLN and ECAN
 */
class IntegratedEvolution {
private:
    MOSESTensorEngine* moses_engine_;
    class PLNTensorEngine* pln_engine_;
    class ECANAttentionSystem* attention_system_;
    
public:
    IntegratedEvolution(MOSESTensorEngine* moses, 
                       PLNTensorEngine* pln,
                       ECANAttentionSystem* attention);
    
    /**
     * @brief Evolve reasoning strategies
     */
    TensorProgram* evolve_reasoning_strategy(const std::vector<Handle>& problem_atoms);
    
    /**
     * @brief Evolve attention allocation policies
     */
    TensorProgram* evolve_attention_policy(const std::vector<Handle>& context_atoms);
    
    /**
     * @brief Co-evolution of reasoning and attention
     */
    std::pair<TensorProgram*, TensorProgram*> co_evolve_systems(
        const std::vector<Handle>& training_atoms, size_t generations = 50);
    
    /**
     * @brief Adaptive cognitive architecture evolution
     */
    void evolve_cognitive_architecture(const std::vector<CognitiveTask>& tasks);
};

} // namespace opencog