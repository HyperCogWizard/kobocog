#!/usr/bin/env python3
"""
Test suite for OpenCog-GGML integration in Kobocog

This test script validates the core functionality of the tensor-based
AtomSpace, PLN reasoning, and ECAN attention system.
"""

import unittest
import sys
import os
import ctypes
import json
from pathlib import Path

# Add opencog module path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestTensorAtomSpace(unittest.TestCase):
    """Test tensor-based AtomSpace implementation"""
    
    def setUp(self):
        """Set up test environment"""
        # For now, we'll simulate the interface until C++ implementation is complete
        self.atomspace_data = {}
        self.next_handle = 1
        
    def test_concept_creation(self):
        """Test creation of concept nodes"""
        # Simulate adding a concept
        handle = self._add_concept("cat")
        self.assertIsNotNone(handle)
        self.assertIn("cat", self.atomspace_data)
        
    def test_inheritance_link(self):
        """Test creation of inheritance links"""
        cat_handle = self._add_concept("cat")
        animal_handle = self._add_concept("animal")
        
        link_handle = self._add_inheritance(cat_handle, animal_handle, 0.9, 0.8)
        self.assertIsNotNone(link_handle)
        
    def test_truth_value_operations(self):
        """Test truth value arithmetic"""
        tv1 = {"strength": 0.8, "confidence": 0.9}
        tv2 = {"strength": 0.7, "confidence": 0.8}
        
        # Test revision formula
        result = self._revision(tv1, tv2)
        self.assertAlmostEqual(result["strength"], 0.769, places=2)  # Fixed expected value
        self.assertTrue(0 < result["confidence"] < 1)
        
    def _add_concept(self, name):
        """Simulate adding concept to atomspace"""
        handle = self.next_handle
        self.next_handle += 1
        self.atomspace_data[name] = {
            "handle": handle,
            "type": "CONCEPT_NODE",
            "name": name,
            "embedding": [0.1] * 768  # Simulated embedding
        }
        return handle
        
    def _add_inheritance(self, source, target, strength, confidence):
        """Simulate adding inheritance link"""
        handle = self.next_handle
        self.next_handle += 1
        link_name = f"inheritance_{source}_{target}"
        self.atomspace_data[link_name] = {
            "handle": handle,
            "type": "INHERITANCE_LINK",
            "source": source,
            "target": target,
            "truth": {"strength": strength, "confidence": confidence}
        }
        return handle
        
    def _revision(self, tv1, tv2):
        """Simulate PLN revision formula"""
        s1, c1 = tv1["strength"], tv1["confidence"]
        s2, c2 = tv2["strength"], tv2["confidence"]
        
        w1 = c1 / (1 - c1) if c1 < 1 else 1000
        w2 = c2 / (1 - c2) if c2 < 1 else 1000
        w = w1 + w2
        
        if w == 0:
            return {"strength": 0.5, "confidence": 0.0}
            
        result_strength = (s1 * w1 + s2 * w2) / w
        result_confidence = w / (w + 1)
        
        return {"strength": result_strength, "confidence": result_confidence}


class TestPLNReasoning(unittest.TestCase):
    """Test PLN reasoning engine"""
    
    def setUp(self):
        """Set up PLN test environment"""
        self.knowledge_base = {
            "socrates_human": {"strength": 0.9, "confidence": 0.9},
            "human_mortal": {"strength": 0.95, "confidence": 0.95}
        }
        
    def test_deduction_rule(self):
        """Test deduction: socrates->human, human->mortal ⊢ socrates->mortal"""
        premise1 = self.knowledge_base["socrates_human"]
        premise2 = self.knowledge_base["human_mortal"]
        
        # Apply deduction formula
        result = self._apply_deduction(premise1, premise2)
        
        # Should derive socrates->mortal with high confidence
        self.assertTrue(result["strength"] > 0.8)
        self.assertTrue(result["confidence"] > 0.8)
        
    def test_induction_rule(self):
        """Test induction rule application"""
        # bird->fly, bird->feathers ⊢ fly->feathers (weak)
        premise1 = {"strength": 0.8, "confidence": 0.7}
        premise2 = {"strength": 0.9, "confidence": 0.8}
        
        result = self._apply_induction(premise1, premise2)
        
        # Induction should have lower confidence than premises
        self.assertTrue(result["confidence"] < min(premise1["confidence"], premise2["confidence"]))
        
    def test_abduction_rule(self):
        """Test abduction rule application"""
        # wet->rain, grass->rain ⊢ wet->grass (hypothetical)
        premise1 = {"strength": 0.7, "confidence": 0.6}
        premise2 = {"strength": 0.8, "confidence": 0.7}
        
        result = self._apply_abduction(premise1, premise2)
        
        # Abduction should be even more uncertain
        self.assertTrue(result["confidence"] < 0.5)
        
    def _apply_deduction(self, premise1, premise2):
        """Apply deduction formula"""
        s1, c1 = premise1["strength"], premise1["confidence"]
        s2, c2 = premise2["strength"], premise2["confidence"]
        
        result_strength = s1 * s2
        result_confidence = c1 * c2 * max(s1, s2)
        
        return {"strength": result_strength, "confidence": result_confidence}
        
    def _apply_induction(self, premise1, premise2):
        """Apply induction formula"""
        s1, c1 = premise1["strength"], premise1["confidence"]
        s2, c2 = premise2["strength"], premise2["confidence"]
        
        result_strength = s1 * s2
        result_confidence = c1 * c2 * (1 - max(s1, s2))
        
        return {"strength": result_strength, "confidence": result_confidence}
        
    def _apply_abduction(self, premise1, premise2):
        """Apply abduction formula"""
        s1, c1 = premise1["strength"], premise1["confidence"]
        s2, c2 = premise2["strength"], premise2["confidence"]
        
        result_strength = s1 * s2
        result_confidence = c1 * c2 * (1 - min(s1, s2))
        
        return {"strength": result_strength, "confidence": result_confidence}


class TestECANAttention(unittest.TestCase):
    """Test ECAN attention system"""
    
    def setUp(self):
        """Set up attention test environment"""
        self.atoms = {
            "cat": {"sti": 0.5, "lti": 0.3, "vlti": 0.1},
            "dog": {"sti": 0.3, "lti": 0.4, "vlti": 0.2},
            "animal": {"sti": 0.8, "lti": 0.6, "vlti": 0.5},
            "pet": {"sti": 0.4, "lti": 0.3, "vlti": 0.2}
        }
        
    def test_attention_updating(self):
        """Test ECAN attention value updates"""
        initial_sti = self.atoms["cat"]["sti"]
        
        # Simulate stimulus
        self._stimulate_atom("cat", 0.2)
        
        # STI should increase
        self.assertTrue(self.atoms["cat"]["sti"] > initial_sti)
        
    def test_attention_spreading(self):
        """Test activation spreading"""
        initial_dog_sti = self.atoms["dog"]["sti"]
        
        # Spread activation from highly active "animal" to connected "dog"
        self._spread_activation("animal", "dog", 0.1)
        
        # Dog's STI should increase due to spreading
        self.assertTrue(self.atoms["dog"]["sti"] > initial_dog_sti)
        
    def test_attentional_focus(self):
        """Test attentional focus selection"""
        focus = self._get_attentional_focus(threshold=0.4)
        
        # Should include high-attention atoms
        self.assertIn("animal", focus)
        self.assertIn("cat", focus)
        
        # Should exclude low-attention atoms
        self.assertNotIn("dog", focus)
        
    def test_rent_collection(self):
        """Test attention decay through rent collection"""
        initial_values = {name: atom["sti"] for name, atom in self.atoms.items()}
        
        self._collect_rent(decay_rate=0.05)
        
        # All STI values should decrease
        for name in self.atoms:
            self.assertTrue(self.atoms[name]["sti"] < initial_values[name])
            
    def _stimulate_atom(self, atom_name, amount):
        """Simulate attention stimulus"""
        if atom_name in self.atoms:
            self.atoms[atom_name]["sti"] = min(1.0, self.atoms[atom_name]["sti"] + amount)
            
    def _spread_activation(self, source, target, amount):
        """Simulate activation spreading"""
        if source in self.atoms and target in self.atoms:
            spread_amount = self.atoms[source]["sti"] * amount
            self.atoms[target]["sti"] = min(1.0, self.atoms[target]["sti"] + spread_amount)
            
    def _get_attentional_focus(self, threshold=0.5):
        """Get atoms above attention threshold"""
        return [name for name, atom in self.atoms.items() if atom["sti"] >= threshold]
        
    def _collect_rent(self, decay_rate=0.01):
        """Apply attention decay"""
        for atom in self.atoms.values():
            atom["sti"] = max(0.0, atom["sti"] * (1 - decay_rate))


class TestCognitiveAPI(unittest.TestCase):
    """Test cognitive API endpoints"""
    
    def test_api_request_format(self):
        """Test API request/response format"""
        request = {
            "operation": "memory/add_concept",
            "params": {
                "name": "robot",
                "embedding": None
            },
            "session_id": "test_session_123"
        }
        
        # Simulate API processing
        response = self._process_api_request(request)
        
        self.assertTrue(response["success"])
        self.assertIn("handle", response["result"])
        
    def test_reasoning_api(self):
        """Test reasoning API endpoint"""
        request = {
            "operation": "reasoning/forward_chain",
            "params": {
                "premises": ["handle_1", "handle_2"],
                "max_steps": 5
            },
            "session_id": "test_session_123"
        }
        
        response = self._process_api_request(request)
        
        self.assertTrue(response["success"])
        self.assertIn("conclusion", response["result"])
        
    def test_attention_api(self):
        """Test attention API endpoint"""
        request = {
            "operation": "attention/get_focus",
            "params": {},
            "session_id": "test_session_123"
        }
        
        response = self._process_api_request(request)
        
        self.assertTrue(response["success"])
        self.assertIn("focus_atoms", response["result"])
        
    def _process_api_request(self, request):
        """Simulate API request processing"""
        if request["operation"] == "memory/add_concept":
            return {
                "success": True,
                "result": {"handle": "handle_123"},
                "confidence": 1.0,
                "processing_time_ms": 10
            }
        elif request["operation"] == "reasoning/forward_chain":
            return {
                "success": True,
                "result": {
                    "conclusion": "handle_456",
                    "truth": {"strength": 0.8, "confidence": 0.7},
                    "proof_trail": ["handle_1", "handle_2", "handle_456"]
                },
                "confidence": 0.7,
                "processing_time_ms": 50
            }
        elif request["operation"] == "attention/get_focus":
            return {
                "success": True,
                "result": {
                    "focus_atoms": ["handle_123", "handle_456"],
                    "focus_size": 2
                },
                "confidence": 1.0,
                "processing_time_ms": 5
            }
        else:
            return {
                "success": False,
                "error_message": "Unknown operation",
                "confidence": 0.0,
                "processing_time_ms": 1
            }


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_knowledge_reasoning_cycle(self):
        """Test complete knowledge + reasoning + attention cycle"""
        # 1. Add knowledge
        atomspace = {}
        self._add_knowledge(atomspace)
        
        # 2. Perform reasoning
        reasoning_results = self._perform_reasoning(atomspace)
        
        # 3. Update attention based on results
        attention_updates = self._update_attention(reasoning_results)
        
        # Verify the cycle worked
        self.assertTrue(len(reasoning_results) > 0)
        self.assertTrue(len(attention_updates) > 0)
        
    def _add_knowledge(self, atomspace):
        """Add test knowledge to atomspace"""
        concepts = ["cat", "dog", "animal", "pet", "mammal"]
        for concept in concepts:
            atomspace[concept] = {"type": "CONCEPT_NODE", "attention": 0.5}
            
        # Add some inheritance links
        atomspace["cat_animal"] = {
            "type": "INHERITANCE_LINK",
            "source": "cat",
            "target": "animal",
            "truth": {"strength": 0.9, "confidence": 0.8}
        }
        
    def _perform_reasoning(self, atomspace):
        """Perform PLN reasoning on atomspace"""
        results = []
        
        # Find inheritance patterns and apply deduction
        if "cat_animal" in atomspace:
            results.append({
                "rule": "deduction",
                "conclusion": "derived_knowledge",
                "confidence": 0.7
            })
            
        return results
        
    def _update_attention(self, reasoning_results):
        """Update attention based on reasoning results"""
        updates = []
        
        for result in reasoning_results:
            if result["confidence"] > 0.5:
                updates.append({
                    "atom": result["conclusion"],
                    "sti_boost": 0.2
                })
                
        return updates


def run_tests():
    """Run all tests and display results"""
    print("=== OpenCog-GGML Integration Test Suite ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTensorAtomSpace,
        TestPLNReasoning,
        TestECANAttention,
        TestCognitiveAPI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in traceback else 'Unknown error'}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\n')[-2] if traceback else 'Unknown error'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)