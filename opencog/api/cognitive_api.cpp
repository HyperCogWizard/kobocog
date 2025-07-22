/**
 * @file cognitive_api.cpp
 * @brief Implementation of cognitive API for Kobocog integration
 */

#include "cognitive_api.h"
#include "../vendor/nlohmann/json.hpp"
#include <chrono>
#include <sstream>
#include <fstream>
#include <random>

namespace opencog {

// CognitiveAPI implementation

CognitiveAPI::CognitiveAPI() : session_timeout_minutes_(30) {
    stats_ = {};
}

CognitiveAPI::~CognitiveAPI() = default;

bool CognitiveAPI::initialize(size_t memory_size) {
    try {
        // Initialize AtomSpace
        atomspace_ = std::make_unique<TensorAtomSpace>(memory_size);
        
        // Initialize PLN engine
        pln_engine_ = std::make_unique<PLNTensorEngine>(atomspace_.get());
        
        // Initialize attention system
        attention_system_ = std::make_unique<ECANAttentionSystem>(atomspace_.get());
        attention_system_->initialize();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize cognitive API: " << e.what() << std::endl;
        return false;
    }
}

CognitiveResponse CognitiveAPI::handle_request(const CognitiveRequest& request) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Validate session
    if (!validate_session(request.session_id)) {
        return create_error_response("Invalid or expired session");
    }
    
    CognitiveResponse response;
    
    try {
        if (request.operation.substr(0, 6) == "memory") {
            response = handle_memory_operation(request.params, request.session_id);
        } else if (request.operation.substr(0, 9) == "reasoning") {
            response = handle_reasoning_operation(request.params, request.session_id);
        } else if (request.operation.substr(0, 9) == "attention") {
            response = handle_attention_operation(request.params, request.session_id);
        } else if (request.operation == "status") {
            response = get_system_status(request.session_id);
        } else {
            response = create_error_response("Unknown operation: " + request.operation);
        }
    } catch (const std::exception& e) {
        response = create_error_response("Operation failed: " + std::string(e.what()));
    }
    
    // Calculate processing time
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    response.processing_time_ms = duration.count();
    
    // Update statistics
    update_stats(request.operation, response.processing_time_ms, response.success);
    
    return response;
}

CognitiveResponse CognitiveAPI::handle_memory_operation(const std::string& params, 
                                                       const std::string& session_id) {
    // Parse operation type from params
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return create_error_response("Invalid JSON parameters");
    }
    
    std::string op_type = root.get("operation", "").asString();
    
    if (op_type == "add_concept") {
        std::string result = MemoryEndpoints::add_concept(atomspace_.get(), params);
        return create_success_response(result);
    } else if (op_type == "add_inheritance") {
        std::string result = MemoryEndpoints::add_inheritance(atomspace_.get(), params);
        return create_success_response(result);
    } else if (op_type == "query_concept") {
        std::string result = MemoryEndpoints::query_concept(atomspace_.get(), params);
        return create_success_response(result);
    } else if (op_type == "get_related") {
        std::string result = MemoryEndpoints::get_related_atoms(atomspace_.get(), params);
        return create_success_response(result);
    } else if (op_type == "get_stats") {
        std::string result = MemoryEndpoints::get_memory_stats(atomspace_.get());
        return create_success_response(result);
    } else {
        return create_error_response("Unknown memory operation: " + op_type);
    }
}

CognitiveResponse CognitiveAPI::handle_reasoning_operation(const std::string& params,
                                                          const std::string& session_id) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return create_error_response("Invalid JSON parameters");
    }
    
    std::string op_type = root.get("operation", "").asString();
    
    if (op_type == "forward_chain") {
        std::string result = ReasoningEndpoints::forward_chain(pln_engine_.get(), params);
        return create_success_response(result);
    } else if (op_type == "backward_chain") {
        std::string result = ReasoningEndpoints::backward_chain(pln_engine_.get(), params);
        return create_success_response(result);
    } else if (op_type == "apply_rule") {
        std::string result = ReasoningEndpoints::apply_rule(pln_engine_.get(), params);
        return create_success_response(result);
    } else if (op_type == "evaluate_truth") {
        std::string result = ReasoningEndpoints::evaluate_truth(pln_engine_.get(), params);
        return create_success_response(result);
    } else {
        return create_error_response("Unknown reasoning operation: " + op_type);
    }
}

CognitiveResponse CognitiveAPI::handle_attention_operation(const std::string& params,
                                                          const std::string& session_id) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return create_error_response("Invalid JSON parameters");
    }
    
    std::string op_type = root.get("operation", "").asString();
    
    if (op_type == "get_focus") {
        std::string result = AttentionEndpoints::get_focus(attention_system_.get());
        return create_success_response(result);
    } else if (op_type == "update_focus") {
        std::string result = AttentionEndpoints::update_focus(attention_system_.get(), params);
        return create_success_response(result);
    } else if (op_type == "spread_activation") {
        std::string result = AttentionEndpoints::spread_activation(attention_system_.get(), params);
        return create_success_response(result);
    } else if (op_type == "get_stats") {
        std::string result = AttentionEndpoints::get_attention_stats(attention_system_.get());
        return create_success_response(result);
    } else {
        return create_error_response("Unknown attention operation: " + op_type);
    }
}

CognitiveResponse CognitiveAPI::get_system_status(const std::string& session_id) {
    Json::Value status;
    
    if (atomspace_) {
        auto atomspace_stats = atomspace_->get_stats();
        status["atomspace"]["num_atoms"] = static_cast<int>(atomspace_stats.num_atoms);
        status["atomspace"]["num_nodes"] = static_cast<int>(atomspace_stats.num_nodes);
        status["atomspace"]["num_links"] = static_cast<int>(atomspace_stats.num_links);
        status["atomspace"]["memory_used"] = static_cast<int>(atomspace_stats.memory_used);
        status["atomspace"]["avg_attention"] = atomspace_stats.avg_attention;
    }
    
    if (pln_engine_) {
        auto pln_stats = pln_engine_->get_statistics();
        status["reasoning"]["total_inferences"] = static_cast<int>(pln_stats.total_inferences);
        status["reasoning"]["successful_inferences"] = static_cast<int>(pln_stats.successful_inferences);
        status["reasoning"]["avg_confidence"] = pln_stats.avg_confidence;
    }
    
    if (attention_system_) {
        auto attention_stats = attention_system_->get_statistics();
        status["attention"]["total_cycles"] = static_cast<int>(attention_stats.total_cycles);
        status["attention"]["avg_focus_size"] = attention_stats.avg_focus_size;
        status["attention"]["efficiency"] = attention_stats.attention_efficiency;
    }
    
    status["api"]["total_requests"] = static_cast<int>(stats_.total_requests);
    status["api"]["successful_requests"] = static_cast<int>(stats_.successful_requests);
    status["api"]["avg_processing_time"] = stats_.avg_processing_time;
    
    Json::StreamWriterBuilder builder;
    std::string result = Json::writeString(builder, status);
    
    return create_success_response(result);
}

// Session management

std::string CognitiveAPI::create_session() {
    // Generate random session ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000000, 9999999);
    
    std::string session_id = "session_" + std::to_string(dis(gen));
    sessions_[session_id] = std::chrono::steady_clock::now();
    
    return session_id;
}

bool CognitiveAPI::validate_session(const std::string& session_id) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false; // Session not found
    }
    
    // Check if session has expired
    auto now = std::chrono::steady_clock::now();
    auto session_age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second);
    
    if (session_age.count() > session_timeout_minutes_) {
        sessions_.erase(it);
        return false; // Session expired
    }
    
    // Update session timestamp
    it->second = now;
    return true;
}

void CognitiveAPI::cleanup_expired_sessions() {
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        auto session_age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second);
        if (session_age.count() > session_timeout_minutes_) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

// Utility functions

std::string CognitiveAPI::handle_to_string(Handle handle) {
    return std::to_string(handle);
}

Handle CognitiveAPI::string_to_handle(const std::string& handle_str) {
    try {
        return std::stoull(handle_str);
    } catch (...) {
        return INVALID_HANDLE;
    }
}

std::string CognitiveAPI::truth_value_to_json(const TruthValue& tv) {
    Json::Value json_tv;
    json_tv["strength"] = tv.strength;
    json_tv["confidence"] = tv.confidence;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, json_tv);
}

TruthValue CognitiveAPI::json_to_truth_value(const std::string& json_str) {
    Json::Value root;
    Json::Reader reader;
    
    if (reader.parse(json_str, root)) {
        return TruthValue(
            root.get("strength", 0.5f).asFloat(),
            root.get("confidence", 0.5f).asFloat()
        );
    }
    
    return TruthValue();
}

CognitiveResponse CognitiveAPI::create_error_response(const std::string& error_msg) {
    return {false, "", error_msg, 0.0f, 0};
}

CognitiveResponse CognitiveAPI::create_success_response(const std::string& result, float confidence) {
    return {true, result, "", confidence, 0};
}

void CognitiveAPI::update_stats(const std::string& operation, int processing_time, bool success) {
    stats_.total_requests++;
    if (success) stats_.successful_requests++;
    
    stats_.avg_processing_time = (stats_.avg_processing_time * (stats_.total_requests - 1) + processing_time) / stats_.total_requests;
    stats_.operation_counts[operation]++;
}

CognitiveAPI::APIStats CognitiveAPI::get_statistics() const {
    return stats_;
}

// MemoryEndpoints implementation

std::string MemoryEndpoints::add_concept(TensorAtomSpace* atomspace, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    std::string name = root.get("name", "").asString();
    if (name.empty()) {
        return R"({"error": "Missing concept name"})";
    }
    
    Handle handle = atomspace->add_concept(name);
    
    Json::Value result;
    result["handle"] = std::to_string(handle);
    result["name"] = name;
    result["type"] = "ConceptNode";
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

std::string MemoryEndpoints::add_inheritance(TensorAtomSpace* atomspace, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    std::string source_str = root.get("source", "").asString();
    std::string target_str = root.get("target", "").asString();
    
    Handle source = std::stoull(source_str);
    Handle target = std::stoull(target_str);
    
    float strength = root.get("strength", 0.5f).asFloat();
    float confidence = root.get("confidence", 0.5f).asFloat();
    
    Handle handle = atomspace->add_inheritance(source, target, TruthValue(strength, confidence));
    
    Json::Value result;
    result["handle"] = std::to_string(handle);
    result["source"] = source_str;
    result["target"] = target_str;
    result["type"] = "InheritanceLink";
    result["truth"]["strength"] = strength;
    result["truth"]["confidence"] = confidence;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

std::string MemoryEndpoints::query_concept(TensorAtomSpace* atomspace, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    std::string name = root.get("name", "").asString();
    Handle handle = atomspace->get_handle(name);
    
    Json::Value result;
    if (handle != INVALID_HANDLE) {
        TensorAtom* atom = atomspace->get_atom(handle);
        result["found"] = true;
        result["handle"] = std::to_string(handle);
        result["name"] = atom->name;
        result["attention"]["sti"] = atom->attention.sti;
        result["attention"]["lti"] = atom->attention.lti;
        result["attention"]["vlti"] = atom->attention.vlti;
    } else {
        result["found"] = false;
    }
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

std::string MemoryEndpoints::get_related_atoms(TensorAtomSpace* atomspace, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    std::string handle_str = root.get("handle", "").asString();
    Handle handle = std::stoull(handle_str);
    
    TensorAtom* atom = atomspace->get_atom(handle);
    if (!atom) {
        return R"({"error": "Atom not found"})";
    }
    
    Json::Value result;
    result["handle"] = handle_str;
    
    Json::Value incoming(Json::arrayValue);
    for (Handle h : atom->incoming) {
        incoming.append(std::to_string(h));
    }
    result["incoming"] = incoming;
    
    Json::Value outgoing(Json::arrayValue);
    for (Handle h : atom->outgoing) {
        outgoing.append(std::to_string(h));
    }
    result["outgoing"] = outgoing;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

std::string MemoryEndpoints::get_memory_stats(TensorAtomSpace* atomspace) {
    auto stats = atomspace->get_stats();
    
    Json::Value result;
    result["num_atoms"] = static_cast<int>(stats.num_atoms);
    result["num_nodes"] = static_cast<int>(stats.num_nodes);
    result["num_links"] = static_cast<int>(stats.num_links);
    result["memory_used"] = static_cast<int>(stats.memory_used);
    result["avg_attention"] = stats.avg_attention;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

// ReasoningEndpoints implementation

std::string ReasoningEndpoints::forward_chain(PLNTensorEngine* pln, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    std::vector<Handle> premises;
    Json::Value premise_array = root.get("premises", Json::arrayValue);
    for (const auto& premise : premise_array) {
        premises.push_back(std::stoull(premise.asString()));
    }
    
    int max_steps = root.get("max_steps", 10).asInt();
    
    PLNResult result = pln->forward_chain(premises, max_steps);
    
    Json::Value json_result;
    json_result["success"] = result.success;
    json_result["conclusion"] = std::to_string(result.conclusion);
    json_result["truth"]["strength"] = result.truth.strength;
    json_result["truth"]["confidence"] = result.truth.confidence;
    json_result["confidence"] = result.confidence;
    
    Json::Value proof_trail(Json::arrayValue);
    for (Handle h : result.proof_trail) {
        proof_trail.append(std::to_string(h));
    }
    json_result["proof_trail"] = proof_trail;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, json_result);
}

std::string ReasoningEndpoints::backward_chain(PLNTensorEngine* pln, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    Handle target = std::stoull(root.get("target", "0").asString());
    int max_steps = root.get("max_steps", 10).asInt();
    
    PLNResult result = pln->backward_chain(target, max_steps);
    
    Json::Value json_result;
    json_result["success"] = result.success;
    json_result["conclusion"] = std::to_string(result.conclusion);
    json_result["truth"]["strength"] = result.truth.strength;
    json_result["truth"]["confidence"] = result.truth.confidence;
    json_result["confidence"] = result.confidence;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, json_result);
}

std::string ReasoningEndpoints::apply_rule(PLNTensorEngine* pln, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    int rule_type = root.get("rule_type", 0).asInt();
    std::vector<Handle> premises;
    Json::Value premise_array = root.get("premises", Json::arrayValue);
    for (const auto& premise : premise_array) {
        premises.push_back(std::stoull(premise.asString()));
    }
    
    PLNResult result = pln->apply_rule(static_cast<PLNRuleType>(rule_type), premises);
    
    Json::Value json_result;
    json_result["success"] = result.success;
    json_result["conclusion"] = std::to_string(result.conclusion);
    json_result["truth"]["strength"] = result.truth.strength;
    json_result["truth"]["confidence"] = result.truth.confidence;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, json_result);
}

std::string ReasoningEndpoints::evaluate_truth(PLNTensorEngine* pln, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    Handle atom = std::stoull(root.get("atom", "0").asString());
    TruthValue tv = pln->evaluate_truth(atom);
    
    Json::Value result;
    result["strength"] = tv.strength;
    result["confidence"] = tv.confidence;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

// AttentionEndpoints implementation

std::string AttentionEndpoints::get_focus(ECANAttentionSystem* attention) {
    auto focus_atoms = attention->get_focus();
    
    Json::Value result;
    Json::Value atoms(Json::arrayValue);
    
    for (Handle atom : focus_atoms) {
        atoms.append(std::to_string(atom));
    }
    
    result["focus_atoms"] = atoms;
    result["focus_size"] = static_cast<int>(focus_atoms.size());
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

std::string AttentionEndpoints::update_focus(ECANAttentionSystem* attention, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    Handle atom = std::stoull(root.get("atom", "0").asString());
    float intensity = root.get("intensity", 1.0f).asFloat();
    
    attention->focus_on_atom(atom, intensity);
    
    return R"({"success": true})";
}

std::string AttentionEndpoints::spread_activation(ECANAttentionSystem* attention, const std::string& params) {
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(params, root)) {
        return R"({"error": "Invalid JSON"})";
    }
    
    Handle atom = std::stoull(root.get("atom", "0").asString());
    float amount = root.get("amount", 0.1f).asFloat();
    
    attention->spread_activation(atom, amount);
    
    return R"({"success": true})";
}

std::string AttentionEndpoints::get_attention_stats(ECANAttentionSystem* attention) {
    auto stats = attention->get_statistics();
    
    Json::Value result;
    result["total_cycles"] = static_cast<int>(stats.total_cycles);
    result["avg_focus_size"] = stats.avg_focus_size;
    result["attention_efficiency"] = stats.attention_efficiency;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, result);
}

// Global instance management
std::unique_ptr<CognitiveAPI> KobocogIntegration::instance_ = nullptr;
bool KobocogIntegration::initialized_ = false;

bool KobocogIntegration::initialize_cognitive_api() {
    if (initialized_) return true;
    
    instance_ = std::make_unique<CognitiveAPI>();
    bool success = instance_->initialize();
    
    if (success) {
        initialized_ = true;
    } else {
        instance_.reset();
    }
    
    return success;
}

CognitiveAPI* KobocogIntegration::get_instance() {
    return instance_.get();
}

void KobocogIntegration::shutdown() {
    instance_.reset();
    initialized_ = false;
}

} // namespace opencog

// C interface implementation

extern "C" {

int opencog_init(size_t memory_size) {
    return opencog::KobocogIntegration::initialize_cognitive_api() ? 1 : 0;
}

const char* opencog_handle_request(const char* request_json) {
    static std::string response_buffer;
    
    auto* api = opencog::KobocogIntegration::get_instance();
    if (!api) {
        response_buffer = R"({"success": false, "error": "OpenCog not initialized"})";
        return response_buffer.c_str();
    }
    
    // Parse request
    opencog::CognitiveRequest request;
    Json::Value root;
    Json::Reader reader;
    
    if (reader.parse(request_json, root)) {
        request.operation = root.get("operation", "").asString();
        request.params = Json::writeString(Json::StreamWriterBuilder(), root.get("params", Json::Value()));
        request.session_id = root.get("session_id", "default").asString();
    } else {
        response_buffer = R"({"success": false, "error": "Invalid JSON request"})";
        return response_buffer.c_str();
    }
    
    // Handle request
    auto response = api->handle_request(request);
    
    // Format response
    Json::Value json_response;
    json_response["success"] = response.success;
    json_response["result"] = response.result;
    json_response["error_message"] = response.error_message;
    json_response["confidence"] = response.confidence;
    json_response["processing_time_ms"] = response.processing_time_ms;
    
    response_buffer = Json::writeString(Json::StreamWriterBuilder(), json_response);
    return response_buffer.c_str();
}

const char* opencog_get_status() {
    static std::string status_buffer;
    
    auto* api = opencog::KobocogIntegration::get_instance();
    if (!api) {
        status_buffer = R"({"initialized": false})";
        return status_buffer.c_str();
    }
    
    auto response = api->get_system_status("system");
    status_buffer = response.result;
    return status_buffer.c_str();
}

void opencog_shutdown() {
    opencog::KobocogIntegration::shutdown();
}

}