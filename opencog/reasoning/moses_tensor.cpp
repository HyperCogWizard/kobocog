/**
 * @file moses_tensor.cpp
 * @brief Implementation of MOSES evolutionary program synthesis using GGML tensors
 */

#include "moses_tensor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>

namespace opencog {

// ProgramPopulation implementation

ProgramPopulation::ProgramPopulation(struct ggml_context* ctx, size_t max_size)
    : max_size_(max_size), ctx_(ctx) {
    programs_.reserve(max_size);
}

ProgramPopulation::~ProgramPopulation() = default;

void ProgramPopulation::add_program(std::unique_ptr<TensorProgram> program) {
    if (programs_.size() < max_size_) {
        programs_.push_back(std::move(program));
    } else {
        // Replace worst program if new one is better
        auto worst_it = std::min_element(programs_.begin(), programs_.end(),
            [](const auto& a, const auto& b) { return a->fitness < b->fitness; });
        
        if (program->fitness > (*worst_it)->fitness) {
            *worst_it = std::move(program);
        }
    }
}

void ProgramPopulation::remove_program(size_t index) {
    if (index < programs_.size()) {
        programs_.erase(programs_.begin() + index);
    }
}

TensorProgram* ProgramPopulation::get_program(size_t index) const {
    return (index < programs_.size()) ? programs_[index].get() : nullptr;
}

void ProgramPopulation::sort_by_fitness() {
    std::sort(programs_.begin(), programs_.end(),
              [](const auto& a, const auto& b) { return a->fitness > b->fitness; });
}

std::vector<TensorProgram*> ProgramPopulation::get_elite(size_t count) const {
    std::vector<TensorProgram*> elite;
    size_t actual_count = std::min(count, programs_.size());
    
    // Assume already sorted by fitness
    for (size_t i = 0; i < actual_count; ++i) {
        elite.push_back(programs_[i].get());
    }
    
    return elite;
}

std::vector<TensorProgram*> ProgramPopulation::tournament_selection(size_t tournament_size, size_t count) {
    std::vector<TensorProgram*> selected;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, programs_.size() - 1);
    
    for (size_t i = 0; i < count; ++i) {
        TensorProgram* best = nullptr;
        float best_fitness = -std::numeric_limits<float>::infinity();
        
        // Run tournament
        for (size_t j = 0; j < tournament_size; ++j) {
            size_t idx = dist(gen);
            if (programs_[idx]->fitness > best_fitness) {
                best = programs_[idx].get();
                best_fitness = programs_[idx]->fitness;
            }
        }
        
        if (best) {
            selected.push_back(best);
        }
    }
    
    return selected;
}

void ProgramPopulation::clear_weak_programs(float fitness_threshold) {
    programs_.erase(
        std::remove_if(programs_.begin(), programs_.end(),
                      [fitness_threshold](const auto& program) {
                          return program->fitness < fitness_threshold;
                      }),
        programs_.end());
}

float ProgramPopulation::get_avg_fitness() const {
    if (programs_.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (const auto& program : programs_) {
        sum += program->fitness;
    }
    return sum / programs_.size();
}

float ProgramPopulation::get_best_fitness() const {
    if (programs_.empty()) return 0.0f;
    
    float best = programs_[0]->fitness;
    for (const auto& program : programs_) {
        best = std::max(best, program->fitness);
    }
    return best;
}

size_t ProgramPopulation::get_avg_complexity() const {
    if (programs_.empty()) return 0;
    
    size_t sum = 0;
    for (const auto& program : programs_) {
        sum += program->complexity;
    }
    return sum / programs_.size();
}

// MOSESTensorEngine implementation

MOSESTensorEngine::MOSESTensorEngine(TensorAtomSpace* atomspace)
    : atomspace_(atomspace), ctx_(atomspace->get_context()),
      rng_(std::random_device{}()), uniform_dist_(0.0f, 1.0f), normal_dist_(0.0f, 1.0f) {
    
    population_ = std::make_unique<ProgramPopulation>(ctx_, params_.population_size);
    stats_ = {};
}

MOSESTensorEngine::~MOSESTensorEngine() = default;

void MOSESTensorEngine::set_parameters(const MOSESParams& params) {
    params_ = params;
    
    // Resize population if needed
    if (population_->size() > params_.population_size) {
        // Keep only the best programs
        population_->sort_by_fitness();
        while (population_->size() > params_.population_size) {
            population_->remove_program(population_->size() - 1);
        }
    }
}

void MOSESTensorEngine::initialize_population(const std::vector<AtomType>& allowed_types, size_t max_depth) {
    // Clear existing population
    population_ = std::make_unique<ProgramPopulation>(ctx_, params_.population_size);
    
    // Generate random programs
    for (size_t i = 0; i < params_.population_size; ++i) {
        auto program = generate_random_program(allowed_types, max_depth);
        if (program) {
            population_->add_program(std::move(program));
        }
    }
    
    std::cout << "Initialized population with " << population_->size() << " programs" << std::endl;
}

TensorProgram* MOSESTensorEngine::evolve(FitnessFunction fitness_func, size_t max_generations) {
    if (max_generations == 0) {
        max_generations = params_.max_generations;
    }
    
    TensorProgram* best_program = nullptr;
    float best_fitness = -std::numeric_limits<float>::infinity();
    
    for (size_t generation = 0; generation < max_generations; ++generation) {
        // Evaluate all programs
        for (size_t i = 0; i < population_->size(); ++i) {
            TensorProgram* program = population_->get_program(i);
            if (program) {
                program->fitness = evaluate_program(*program, fitness_func);
            }
        }
        
        // Sort by fitness
        population_->sort_by_fitness();
        
        // Track best program
        TensorProgram* current_best = population_->get_program(0);
        if (current_best && current_best->fitness > best_fitness) {
            best_fitness = current_best->fitness;
            best_program = current_best;
        }
        
        // Create next generation
        auto elite = population_->get_elite(static_cast<size_t>(params_.population_size * params_.elite_fraction));
        
        // Clear population keeping elite
        auto new_population = std::make_unique<ProgramPopulation>(ctx_, params_.population_size);
        for (auto* elite_program : elite) {
            // Deep copy elite programs
            auto copy = std::make_unique<TensorProgram>(*elite_program);
            new_population->add_program(std::move(copy));
        }
        
        // Generate offspring
        while (new_population->size() < params_.population_size) {
            if (uniform_dist_(rng_) < params_.crossover_rate) {
                // Crossover
                auto parents = population_->tournament_selection(params_.tournament_size, 2);
                if (parents.size() >= 2) {
                    auto offspring = crossover(*parents[0], *parents[1]);
                    if (offspring) {
                        // Apply mutation
                        if (uniform_dist_(rng_) < params_.mutation_rate) {
                            mutate(*offspring);
                        }
                        new_population->add_program(std::move(offspring));
                        stats_.successful_crossovers++;
                    }
                }
            } else {
                // Mutation only
                auto parents = population_->tournament_selection(params_.tournament_size, 1);
                if (!parents.empty()) {
                    auto offspring = std::make_unique<TensorProgram>(*parents[0]);
                    mutate(*offspring);
                    new_population->add_program(std::move(offspring));
                    stats_.successful_mutations++;
                }
            }
        }
        
        population_ = std::move(new_population);
        
        // Update statistics
        update_statistics(best_fitness);
        stats_.generation_count++;
        
        // Adaptive parameter adjustment
        if (params_.adaptive_rates) {
            float fitness_improvement = (generation > 0) ? 
                best_fitness - stats_.avg_fitness_history[(generation - 1) % 100] : 0.0f;
            adapt_mutation_rate(fitness_improvement);
        }
        
        // Progress reporting
        if (generation % 10 == 0) {
            std::cout << "Generation " << generation << ": Best fitness = " << best_fitness 
                      << ", Avg fitness = " << population_->get_avg_fitness() << std::endl;
        }
    }
    
    return best_program;
}

float MOSESTensorEngine::evaluate_program(TensorProgram& program, FitnessFunction fitness_func) {
    float fitness = fitness_func(program, atomspace_);
    
    // Apply complexity penalty
    float penalty = complexity_penalty(program.complexity);
    fitness -= penalty;
    
    stats_.evaluations_count++;
    return fitness;
}

std::unique_ptr<TensorProgram> MOSESTensorEngine::crossover(const TensorProgram& parent1, const TensorProgram& parent2) {
    auto offspring = std::make_unique<TensorProgram>();
    
    // Select random crossover point
    float crossover_point = uniform_dist_(rng_);
    
    // Create hybrid program structure
    offspring->root_atom = crossover_subtrees(parent1.root_atom, parent2.root_atom, crossover_point);
    
    if (offspring->root_atom != INVALID_HANDLE) {
        // Build subprogram list
        build_program_tree(*offspring, offspring->root_atom, 0, params_.max_program_size);
        
        // Encode as tensor
        offspring->representation = encode_program(*offspring);
        offspring->complexity = calculate_complexity(*offspring);
        
        return offspring;
    }
    
    return nullptr;
}

void MOSESTensorEngine::mutate(TensorProgram& program, float mutation_strength) {
    // Select random mutation type
    float mutation_type = uniform_dist_(rng_);
    
    if (mutation_type < 0.3f) {
        // Point mutation: replace random atom
        if (!program.subprograms.empty()) {
            size_t idx = static_cast<size_t>(uniform_dist_(rng_) * program.subprograms.size());
            std::vector<AtomType> types = {AtomType::CONCEPT_NODE, AtomType::PREDICATE_NODE, AtomType::INHERITANCE_LINK};
            Handle new_atom = create_random_atom(types);
            program.subprograms[idx] = new_atom;
        }
    } else if (mutation_type < 0.6f) {
        // Subtree mutation: replace random subtree
        mutate_subtree(program.root_atom, mutation_strength);
    } else {
        // Parameter mutation: modify tensor representation
        if (program.representation && program.representation->data) {
            float* data = (float*)program.representation->data;
            size_t size = ggml_nelements(program.representation);
            
            for (size_t i = 0; i < size; ++i) {
                if (uniform_dist_(rng_) < params_.mutation_rate) {
                    data[i] += normal_dist_(rng_) * mutation_strength * 0.1f;
                }
            }
        }
    }
    
    // Recalculate complexity
    program.complexity = calculate_complexity(program);
}

TensorProgram* MOSESTensorEngine::get_best_program() const {
    if (population_->size() == 0) return nullptr;
    
    population_->sort_by_fitness();
    return population_->get_program(0);
}

MOSESTensorEngine::MOSESStats MOSESTensorEngine::get_statistics() const {
    return stats_;
}

void MOSESTensorEngine::reset() {
    population_ = std::make_unique<ProgramPopulation>(ctx_, params_.population_size);
    stats_ = {};
}

// Private methods

std::unique_ptr<TensorProgram> MOSESTensorEngine::generate_random_program(
    const std::vector<AtomType>& allowed_types, size_t max_depth) {
    
    auto program = std::make_unique<TensorProgram>();
    
    // Create random root atom
    program->root_atom = create_random_atom(allowed_types);
    if (program->root_atom == INVALID_HANDLE) {
        return nullptr;
    }
    
    // Build program tree
    build_program_tree(*program, program->root_atom, 0, max_depth);
    
    // Encode as tensor
    program->representation = encode_program(*program);
    program->complexity = calculate_complexity(*program);
    
    return program;
}

Handle MOSESTensorEngine::create_random_atom(const std::vector<AtomType>& allowed_types) {
    if (allowed_types.empty()) return INVALID_HANDLE;
    
    // Select random type
    size_t type_idx = static_cast<size_t>(uniform_dist_(rng_) * allowed_types.size());
    AtomType type = allowed_types[type_idx];
    
    switch (type) {
        case AtomType::CONCEPT_NODE: {
            std::string name = "concept_" + std::to_string(static_cast<int>(uniform_dist_(rng_) * 1000));
            return atomspace_->add_concept(name);
        }
        case AtomType::PREDICATE_NODE: {
            std::string name = "predicate_" + std::to_string(static_cast<int>(uniform_dist_(rng_) * 1000));
            return atomspace_->add_concept(name); // Simplified: use concepts for predicates
        }
        case AtomType::INHERITANCE_LINK: {
            // Create random inheritance link
            Handle source = create_random_atom({AtomType::CONCEPT_NODE});
            Handle target = create_random_atom({AtomType::CONCEPT_NODE});
            if (source != INVALID_HANDLE && target != INVALID_HANDLE) {
                float strength = uniform_dist_(rng_);
                float confidence = uniform_dist_(rng_);
                return atomspace_->add_inheritance(source, target, TruthValue(strength, confidence));
            }
            break;
        }
        default:
            break;
    }
    
    return INVALID_HANDLE;
}

void MOSESTensorEngine::build_program_tree(TensorProgram& program, Handle root, size_t depth, size_t max_depth) {
    if (depth >= max_depth || program.subprograms.size() >= params_.max_program_size) {
        return;
    }
    
    program.subprograms.push_back(root);
    
    // For links, recursively add connected atoms
    TensorAtom* atom = atomspace_->get_atom(root);
    if (atom) {
        for (Handle connected : atom->outgoing) {
            build_program_tree(program, connected, depth + 1, max_depth);
        }
    }
}

Handle MOSESTensorEngine::crossover_subtrees(Handle tree1, Handle tree2, float crossover_point) {
    // Simplified crossover: return tree1 or tree2 based on crossover point
    return (uniform_dist_(rng_) < crossover_point) ? tree1 : tree2;
}

void MOSESTensorEngine::mutate_subtree(Handle& root, float mutation_strength) {
    // Simplified mutation: randomly replace with new atom
    if (uniform_dist_(rng_) < mutation_strength) {
        std::vector<AtomType> types = {AtomType::CONCEPT_NODE, AtomType::PREDICATE_NODE};
        root = create_random_atom(types);
    }
}

struct ggml_tensor* MOSESTensorEngine::encode_program(const TensorProgram& program) {
    // Create fixed-size encoding tensor
    const size_t encoding_size = 1024; // Fixed encoding size
    struct ggml_tensor* encoding = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, encoding_size);
    
    if (encoding && encoding->data) {
        float* data = (float*)encoding->data;
        
        // Initialize with zeros
        std::fill(data, data + encoding_size, 0.0f);
        
        // Encode program structure
        size_t idx = 0;
        for (Handle atom : program.subprograms) {
            if (idx >= encoding_size) break;
            
            // Simple encoding: use handle value normalized
            data[idx] = static_cast<float>(atom) / 1000000.0f;
            idx++;
        }
        
        // Add complexity and fitness information
        if (idx < encoding_size - 2) {
            data[idx] = static_cast<float>(program.complexity) / 100.0f;
            data[idx + 1] = program.fitness;
        }
    }
    
    return encoding;
}

size_t MOSESTensorEngine::calculate_complexity(const TensorProgram& program) {
    return program.subprograms.size();
}

float MOSESTensorEngine::complexity_penalty(size_t complexity) {
    return params_.complexity_penalty * static_cast<float>(complexity);
}

void MOSESTensorEngine::update_statistics(float fitness) {
    size_t idx = stats_.generation_count % 100;
    stats_.avg_fitness_history[idx] = fitness;
    stats_.best_fitness_ever = std::max(stats_.best_fitness_ever, fitness);
}

void MOSESTensorEngine::adapt_mutation_rate(float fitness_improvement) {
    if (fitness_improvement > 0.01f) {
        // Good improvement, reduce mutation rate slightly
        params_.mutation_rate *= 0.95f;
    } else {
        // Poor improvement, increase mutation rate
        params_.mutation_rate *= 1.05f;
    }
    
    // Keep within bounds
    params_.mutation_rate = std::max(0.01f, std::min(0.5f, params_.mutation_rate));
}

void MOSESTensorEngine::adapt_crossover_rate(float population_diversity) {
    // Adjust crossover rate based on population diversity
    if (population_diversity < 0.1f) {
        // Low diversity, increase crossover
        params_.crossover_rate *= 1.05f;
    } else if (population_diversity > 0.8f) {
        // High diversity, can reduce crossover
        params_.crossover_rate *= 0.95f;
    }
    
    params_.crossover_rate = std::max(0.1f, std::min(0.9f, params_.crossover_rate));
}

// MOSESFitnessFunctions implementation

float MOSESFitnessFunctions::boolean_sat_fitness(const TensorProgram& program, TensorAtomSpace* atomspace) {
    // Simplified Boolean SAT fitness
    float fitness = 0.0f;
    
    // Check if program represents a valid Boolean formula
    for (Handle atom : program.subprograms) {
        TensorAtom* tensor_atom = atomspace->get_atom(atom);
        if (tensor_atom && tensor_atom->type == AtomType::INHERITANCE_LINK) {
            // Reward logical structure
            fitness += tensor_atom->truth.strength * tensor_atom->truth.confidence;
        }
    }
    
    return fitness / std::max(1.0f, static_cast<float>(program.subprograms.size()));
}

float MOSESFitnessFunctions::symbolic_regression_fitness(const TensorProgram& program, 
                                                       TensorAtomSpace* atomspace,
                                                       const std::vector<std::pair<float, float>>& data) {
    // Simplified symbolic regression fitness
    float mse = 0.0f;
    
    for (const auto& point : data) {
        float input = point.first;
        float expected = point.second;
        
        // Evaluate program at input (simplified)
        float predicted = 0.0f;
        for (Handle atom : program.subprograms) {
            TensorAtom* tensor_atom = atomspace->get_atom(atom);
            if (tensor_atom) {
                predicted += tensor_atom->truth.strength * input;
            }
        }
        
        float error = predicted - expected;
        mse += error * error;
    }
    
    mse /= data.size();
    return 1.0f / (1.0f + mse); // Convert to fitness (higher is better)
}

// PLNRuleLearner implementation

PLNRuleLearner::PLNRuleLearner(MOSESTensorEngine* engine) : moses_engine_(engine) {}

void PLNRuleLearner::add_training_example(const std::vector<Handle>& premises, Handle conclusion) {
    training_data_.emplace_back(premises, conclusion);
}

TensorProgram* PLNRuleLearner::learn_rule(size_t max_generations) {
    // Set up fitness function for rule learning
    auto fitness_func = [this](const TensorProgram& program, TensorAtomSpace* atomspace) -> float {
        return evaluate_rule_fitness(program);
    };
    
    // Initialize population with logic-oriented atom types
    std::vector<AtomType> logic_types = {
        AtomType::CONCEPT_NODE,
        AtomType::PREDICATE_NODE,
        AtomType::INHERITANCE_LINK,
        AtomType::IMPLICATION_LINK
    };
    
    moses_engine_->initialize_population(logic_types, 5);
    
    // Evolve rule
    return moses_engine_->evolve(fitness_func, max_generations);
}

float PLNRuleLearner::evaluate_rule_fitness(const TensorProgram& rule) {
    float fitness = 0.0f;
    
    // Test rule against training data
    for (const auto& example : training_data_) {
        // Simplified evaluation: check if rule structure matches example
        bool matches = (rule.subprograms.size() >= example.first.size());
        if (matches) {
            fitness += 1.0f;
        }
    }
    
    return fitness / std::max(1.0f, static_cast<float>(training_data_.size()));
}

} // namespace opencog