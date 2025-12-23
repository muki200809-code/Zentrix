```python
# File: advanced_ai_code_engine.py
# PART 1/3: Core Architecture, Abstract Interfaces, and Multi-Model Integration Layer

import os
import json
import asyncio
import logging
import hashlib
import aiohttp
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import base64
import re
from pathlib import Path
import tempfile
import shutil



# ========== Keep ALL the content from your original main.py file ==========
# [Include everything from your main.py - this is the base engine]
# Make sure it ends with the AICodeEngine class definition
# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE ENUMS AND DATA STRUCTURES
# ============================================================================

class ModelProvider(Enum):
    """Supported AI model providers"""
    DEEPSEEK_R1 = "deepseek-r1"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT = "gpt"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"

class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    PHP = "php"
    RUBY = "ruby"
    CSHARP = "csharp"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    DART = "dart"
    SHELL = "shell"

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"

class MediaType(Enum):
    """Supported media types for teaching"""
    VIDEO = "video"
    IMAGE = "image"
    DIAGRAM = "diagram"
    ANIMATION = "animation"
    AUDIO = "audio"
    INTERACTIVE = "interactive"

@dataclass
class CodeGenerationRequest:
    """Enhanced request structure for code generation"""
    prompt: str
    language: CodeLanguage = CodeLanguage.PYTHON
    context: str = ""
    temperature: float = 0.6
    max_tokens: int = 4000
    file_name: Optional[str] = None
    model_provider: Optional[ModelProvider] = None
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    include_explanations: bool = True
    include_tests: bool = False
    security_scan: bool = True
    performance_optimize: bool = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    custom_instructions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeGenerationResponse:
    """Enhanced response structure"""
    code: str
    metadata: Dict[str, Any]
    success: bool
    model_used: str
    provider: ModelProvider
    execution_time: float
    token_usage: Dict[str, int]
    confidence_score: float
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    security_scan_results: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    teaching_resources: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    raw_responses: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking"""
    model_name: str
    provider: ModelProvider
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    token_efficiency: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

# ============================================================================
# ABSTRACT INTERFACES
# ============================================================================

class AIModelInterface(ABC):
    """Abstract interface for all AI models"""
    
    @abstractmethod
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code based on request"""
        pass
    
    @abstractmethod
    async def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze existing code"""
        pass
    
    @abstractmethod
    async def explain_code(self, code: str, language: CodeLanguage, 
                          detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain code with different detail levels"""
        pass
    
    @abstractmethod
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str) -> Dict[str, Any]:
        """Refactor/optimize existing code"""
        pass

class SecurityScannerInterface(ABC):
    """Abstract interface for security scanning"""
    
    @abstractmethod
    async def scan_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        pass
    
    @abstractmethod
    async def scan_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Scan dependencies for vulnerabilities"""
        pass

class PerformanceOptimizerInterface(ABC):
    """Abstract interface for performance optimization"""
    
    @abstractmethod
    async def optimize_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Optimize code for performance"""
        pass
    
    @abstractmethod
    async def benchmark_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Benchmark code performance"""
        pass

class TeachingResourceGenerator(ABC):
    """Abstract interface for teaching resource generation"""
    
    @abstractmethod
    async def generate_explanation(self, concept: str, code: str, 
                                  language: CodeLanguage) -> Dict[str, Any]:
        """Generate concept explanation"""
        pass
    
    @abstractmethod
    async def generate_tutorial(self, topic: str, difficulty: str,
                               language: CodeLanguage) -> Dict[str, Any]:
        """Generate step-by-step tutorial"""
        pass

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class EngineConfig:
    """Central configuration management for the engine"""
    
    def __init__(self):
        self.config = {
            # Model configurations
            "models": {
                "deepseek": {
                    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                    "api_url": os.getenv("DEEPSEEK_URL", "https://api.deepseek.com"),
                    "enabled": True,
                    "default_model": "deepseek-r1",
                    "temperature": 0.6,
                    "max_tokens": 8192,
                    "timeout": 60,
                    "retry_attempts": 3
                },
                "gemini": {
                    "api_key": os.getenv("GOOGLE_API_KEY", ""),
                    "api_url": os.getenv("GOOGLE_URL", "https://generativelanguage.googleapis.com"),
                    "enabled": True,
                    "default_model": "gemini-2.5-pro-exp-03-25",
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "timeout": 60,
                    "retry_attempts": 3,
                    "enable_veo": True,
                    "enable_imagen": True
                },
                "claude": {
                    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                    "api_url": os.getenv("ANTHROPIC_URL", "https://api.anthropic.com"),
                    "enabled": True,
                    "default_model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "timeout": 60,
                    "retry_attempts": 3
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "api_url": os.getenv("OPENAI_URL", "https://api.openai.com"),
                    "enabled": True,
                    "default_model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "timeout": 60,
                    "retry_attempts": 3
                }
            },
            
            # Engine behavior
            "engine": {
                "default_provider": "ensemble",
                "enable_caching": True,
                "cache_ttl": 3600,
                "enable_rate_limiting": True,
                "rate_limit_per_minute": 60,
                "enable_load_balancing": True,
                "max_concurrent_requests": 10,
                "timeout": 120,
                "enable_analytics": True,
                "enable_teaching": True
            },
            
            # Security settings
            "security": {
                "enable_scanning": True,
                "scan_level": "strict",
                "block_dangerous_patterns": True,
                "sanitize_inputs": True,
                "validate_outputs": True,
                "enable_encryption": True,
                "encryption_key": os.getenv("ENGINE_ENCRYPTION_KEY", "")
            },
            
            # Performance settings
            "performance": {
                "enable_optimization": True,
                "optimization_level": "aggressive",
                "enable_parallel_processing": True,
                "max_workers": 4,
                "enable_compression": True,
                "compression_level": 6
            },
            
            # Teaching settings
            "teaching": {
                "enable_explanations": True,
                "explanation_depth": "detailed",
                "enable_examples": True,
                "enable_diagrams": True,
                "enable_videos": False,  # Requires video generation API
                "preferred_format": "markdown"
            }
        }
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration integrity"""
        required_env_vars = ["DEEPSEEK_API_KEY", "GOOGLE_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
    
    def get_model_config(self, provider: str) -> Dict:
        """Get configuration for specific model provider"""
        return self.config["models"].get(provider, {})
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration dynamically"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            logger.info(f"Updated config: {section}.{key} = {value}")

# ============================================================================
# CORE MODEL IMPLEMENTATIONS
# ============================================================================

class DeepSeekR1Model(AIModelInterface):
    """DeepSeek R1 model implementation with reasoning capabilities"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_config = config.get_model_config("deepseek")
        self.session = None
        self.cache = {}
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.model_config.get("timeout", 60))
            )
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code using DeepSeek R1 with reasoning"""
        await self._ensure_session()
        
        start_time = datetime.now()
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if self.config.config["engine"]["enable_caching"] and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now().timestamp() - cached["timestamp"] < self.config.config["engine"]["cache_ttl"]:
                logger.info(f"Using cached response for {cache_key}")
                return cached["response"]
        
        try:
            # Prepare prompt according to DeepSeek R1 specifications
            prompt = self._prepare_deepseek_prompt(request)
            
            headers = {
                "Authorization": f"Bearer {self.model_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": request.temperature if request.temperature else self.model_config["temperature"],
                "max_tokens": min(request.max_tokens, self.model_config["max_tokens"]),
                "stream": False
            }
            
            # Add thinking requirement for DeepSeek R1
            if "r1" in self.model_config["default_model"].lower():
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 2000
                }
            
            url = f"{self.model_config['api_url']}/v1/chat/completions"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Extract code and thinking process
                generated_text = result["choices"][0]["message"]["content"]
                code, thinking = self._extract_code_and_thinking(generated_text)
                
                # Generate teaching resources
                teaching_resources = []
                if request.include_explanations:
                    teaching_resources = await self._generate_teaching_resources(code, thinking, request.language)
                
                # Create response
                code_response = CodeGenerationResponse(
                    code=code,
                    metadata={
                        "thinking_process": thinking,
                        "model": self.model_config["default_model"],
                        "finish_reason": result["choices"][0].get("finish_reason", "stop"),
                        "thinking_tokens": result.get("usage", {}).get("thinking_tokens", 0)
                    },
                    success=True,
                    model_used=self.model_config["default_model"],
                    provider=ModelProvider.DEEPSEEK_R1,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    token_usage=result.get("usage", {}),
                    confidence_score=self._calculate_confidence(code, request),
                    teaching_resources=teaching_resources,
                    raw_responses={"deepseek": result}
                )
                
                # Cache the response
                if self.config.config["engine"]["enable_caching"]:
                    self.cache[cache_key] = {
                        "response": code_response,
                        "timestamp": datetime.now().timestamp()
                    }
                
                return code_response
                
        except Exception as e:
            logger.error(f"DeepSeek R1 generation error: {str(e)}")
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used=self.model_config["default_model"],
                provider=ModelProvider.DEEPSEEK_R1,
                execution_time=(datetime.now() - start_time).total_seconds(),
                token_usage={},
                confidence_score=0.0,
                error=str(e)
            )
    
    def _prepare_deepseek_prompt(self, request: CodeGenerationRequest) -> str:
        """Prepare prompt according to DeepSeek R1 best practices"""
        
        # Base prompt structure for DeepSeek R1
        prompt_parts = []
        
        # Add thinking directive for R1 models
        if "r1" in self.model_config["default_model"].lower():
            prompt_parts.append("Please reason step by step, and put your final answer within \\boxed{}.")
            prompt_parts.append("Start your thinking with <think> tag.")
        
        # Add task description
        prompt_parts.append(f"Task: Generate {request.language.value} code for:")
        prompt_parts.append(request.prompt)
        
        # Add context if provided
        if request.context:
            prompt_parts.append("\nContext:")
            prompt_parts.append(request.context)
        
        # Add language-specific requirements
        prompt_parts.append(f"\nRequirements:")
        prompt_parts.append(f"- Language: {request.language.value}")
        prompt_parts.append(f"- Include comments explaining key concepts")
        prompt_parts.append(f"- Follow best practices for {request.language.value}")
        
        if request.include_tests:
            prompt_parts.append("- Include unit tests")
        
        if request.performance_optimize:
            prompt_parts.append("- Optimize for performance")
        
        # Add security considerations
        if request.security_scan:
            prompt_parts.append("- Ensure code is secure and follows security best practices")
        
        # Add file name context
        if request.file_name:
            prompt_parts.append(f"- File name: {request.file_name}")
        
        # Add custom instructions
        for key, value in request.custom_instructions.items():
            prompt_parts.append(f"- {key}: {value}")
        
        return "\n".join(prompt_parts)
    
    def _extract_code_and_thinking(self, text: str) -> Tuple[str, str]:
        """Extract code and thinking process from DeepSeek R1 response"""
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        thinking = ""
        code = ""
        
        # Extract thinking (content between <think> tags or before code)
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # If no think tags, look for reasoning before code
        if not thinking:
            # Try to find reasoning text before first code block
            if code_blocks:
                code_start = text.find('```')
                thinking = text[:code_start].strip()
        
        # Extract code from code blocks
        if code_blocks:
            # Join all code blocks
            code = '\n\n'.join([block.strip() for block in code_blocks])
        else:
            # Try to find code without backticks
            # Look for function/class definitions
            lines = text.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['def ', 'class ', 'function ', 'const ', 'let ', 'var ', 'import ', 'export ']):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
            
            code = '\n'.join(code_lines)
        
        return code.strip(), thinking.strip()
    
    async def _generate_teaching_resources(self, code: str, thinking: str, 
                                         language: CodeLanguage) -> List[Dict]:
        """Generate teaching resources for the generated code"""
        
        resources = []
        
        # Generate concept explanation
        if thinking:
            resources.append({
                "type": "concept_explanation",
                "title": "Reasoning Process",
                "content": thinking,
                "format": "markdown",
                "difficulty": "intermediate"
            })
        
        # Extract key concepts from code
        concepts = self._extract_concepts_from_code(code, language)
        
        for concept in concepts:
            resources.append({
                "type": "code_concept",
                "title": f"Understanding: {concept['name']}",
                "content": concept['explanation'],
                "format": "markdown",
                "difficulty": concept['difficulty'],
                "code_example": concept['example']
            })
        
        return resources
    
    def _extract_concepts_from_code(self, code: str, language: CodeLanguage) -> List[Dict]:
        """Extract programming concepts from code"""
        
        concepts = []
        
        # Language-specific concept detection
        if language == CodeLanguage.PYTHON:
            # Detect functions
            func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
            for func in func_matches:
                concepts.append({
                    "name": f"Function: {func}",
                    "explanation": f"This function performs a specific task. In Python, functions are defined using the 'def' keyword.",
                    "difficulty": "beginner",
                    "example": f"def {func}(...):"
                })
            
            # Detect classes
            class_matches = re.findall(r'class\s+(\w+)', code)
            for cls in class_matches:
                concepts.append({
                    "name": f"Class: {cls}",
                    "explanation": f"This is a class definition. Classes are blueprints for creating objects in object-oriented programming.",
                    "difficulty": "intermediate",
                    "example": f"class {cls}:"
                })
            
            # Detect imports
            import_matches = re.findall(r'import\s+(\w+)', code)
            for imp in import_matches:
                concepts.append({
                    "name": f"Import: {imp}",
                    "explanation": f"This imports the {imp} module/library for use in the code.",
                    "difficulty": "beginner",
                    "example": f"import {imp}"
                })
        
        return concepts[:5]  # Limit to 5 concepts
    
    def _calculate_confidence(self, code: str, request: CodeGenerationRequest) -> float:
        """Calculate confidence score for generated code"""
        
        score = 0.7  # Base score
        
        # Check code length
        if len(code) > 100:
            score += 0.1
        
        # Check for syntax indicators
        if request.language == CodeLanguage.PYTHON:
            if 'def ' in code or 'class ' in code:
                score += 0.1
            if 'import ' in code:
                score += 0.05
        
        # Check for comments (good practice)
        comment_patterns = [r'#.*', r'""".*?"""', r"'''.*?'''"]
        for pattern in comment_patterns:
            if re.search(pattern, code, re.DOTALL):
                score += 0.05
                break
        
        return min(score, 1.0)
    
    def _generate_cache_key(self, request: CodeGenerationRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": request.prompt[:100],
            "language": request.language.value,
            "context_hash": hashlib.md5(request.context.encode()).hexdigest()[:8] if request.context else "",
            "temperature": request.temperature,
            "model": self.model_config["default_model"]
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze code using DeepSeek R1"""
        await self._ensure_session()
        
        try:
            prompt = f"""
            Analyze the following {language.value} code:
            
            {code}
            
            Please provide analysis covering:
            1. Code quality assessment
            2. Potential bugs or issues
            3. Security vulnerabilities
            4. Performance considerations
            5. Best practices compliance
            6. Improvement suggestions
            
            Format the response as JSON with these keys:
            - quality_score (0-100)
            - issues (list)
            - security_concerns (list)
            - performance_metrics (dict)
            - suggestions (list)
            - complexity_rating (simple/medium/complex)
            """
            
            headers = {
                "Authorization": f"Bearer {self.model_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"}
            }
            
            url = f"{self.model_config['api_url']}/v1/chat/completions"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                analysis_text = result["choices"][0]["message"]["content"]
                
                try:
                    analysis = json.loads(analysis_text)
                except json.JSONDecodeError:
                    # Fallback parsing
                    analysis = self._parse_analysis_text(analysis_text)
                
                return analysis
                
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {str(e)}")
            return {
                "quality_score": 0,
                "issues": ["Analysis failed"],
                "security_concerns": [],
                "performance_metrics": {},
                "suggestions": [],
                "complexity_rating": "unknown",
                "error": str(e)
            }
    
    def _parse_analysis_text(self, text: str) -> Dict[str, Any]:
        """Parse analysis text when JSON parsing fails"""
        
        analysis = {
            "quality_score": 50,
            "issues": [],
            "security_concerns": [],
            "performance_metrics": {},
            "suggestions": [],
            "complexity_rating": "medium"
        }
        
        # Simple text parsing
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            if "quality" in line_lower and "score" in line_lower:
                # Try to extract score
                match = re.search(r'(\d+)/100|\b(\d+)\b', line)
                if match:
                    for group in match.groups():
                        if group and group.isdigit():
                            analysis["quality_score"] = int(group)
                            break
            
            elif "issue" in line_lower or "bug" in line_lower:
                if line.strip() and not line_lower.startswith(("##", "**")):
                    analysis["issues"].append(line.strip())
            
            elif "security" in line_lower:
                if line.strip() and not line_lower.startswith(("##", "**")):
                    analysis["security_concerns"].append(line.strip())
            
            elif "suggestion" in line_lower or "improvement" in line_lower:
                if line.strip() and not line_lower.startswith(("##", "**")):
                    analysis["suggestions"].append(line.strip())
        
        return analysis
    
    async def explain_code(self, code: str, language: CodeLanguage, 
                          detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain code with different detail levels"""
        await self._ensure_session()
        
        try:
            detail_map = {
                "simple": "Explain in simple terms for beginners",
                "detailed": "Provide detailed technical explanation",
                "expert": "Provide expert-level analysis with optimizations"
            }
            
            prompt = f"""
            {detail_map.get(detail_level, "Explain")} the following {language.value} code:
            
            {code}
            
            Include:
            1. Overall purpose
            2. Key functions/methods
            3. Algorithm/flow explanation
            4. Important variables/data structures
            5. Complexity analysis (if applicable)
            
            Format for {detail_level} understanding.
            """
            
            headers = {
                "Authorization": f"Bearer {self.model_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 3000
            }
            
            url = f"{self.model_config['api_url']}/v1/chat/completions"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                explanation = result["choices"][0]["message"]["content"]
                
                return {
                    "explanation": explanation,
                    "detail_level": detail_level,
                    "language": language.value,
                    "code_length": len(code),
                    "estimated_reading_time": len(explanation.split()) // 200  # 200 WPM
                }
                
        except Exception as e:
            logger.error(f"DeepSeek explanation error: {str(e)}")
            return {
                "explanation": f"Failed to generate explanation: {str(e)}",
                "detail_level": detail_level,
                "error": str(e)
            }
    
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str) -> Dict[str, Any]:
        """Refactor/optimize existing code"""
        await self._ensure_session()
        
        try:
            prompt = f"""
            Refactor the following {language.value} code with focus on {optimization_target}:
            
            {code}
            
            Please provide:
            1. Refactored code
            2. Explanation of changes made
            3. Performance improvements (if any)
            4. Readability improvements
            5. Before/after comparison
            
            Optimization target: {optimization_target}
            """
            
            headers = {
                "Authorization": f"Bearer {self.model_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000
            }
            
            url = f"{self.model_config['api_url']}/v1/chat/completions"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                refactored_output = result["choices"][0]["message"]["content"]
                
                # Extract refactored code
                refactored_code = self._extract_refactored_code(refactored_output)
                
                return {
                    "original_code": code,
                    "refactored_code": refactored_code,
                    "optimization_target": optimization_target,
                    "explanation": refactored_output,
                    "estimated_improvement": "Varies based on changes"
                }
                
        except Exception as e:
            logger.error(f"DeepSeek refactoring error: {str(e)}")
            return {
                "original_code": code,
                "refactored_code": code,
                "error": str(e),
                "optimization_target": optimization_target
            }
    
    def _extract_refactored_code(self, text: str) -> str:
        """Extract refactored code from response"""
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        
        if code_blocks:
            # Use the last code block (usually the refactored version)
            return code_blocks[-1].strip()
        
        # Try to find code after keywords
        keywords = ["refactored code:", "optimized code:", "final code:"]
        for keyword in keywords:
            if keyword.lower() in text.lower():
                start_idx = text.lower().find(keyword.lower()) + len(keyword)
                # Take next 100 lines or until next section
                lines = text[start_idx:].split('\n')
                code_lines = []
                for line in lines[:100]:
                    if line.strip() and not line.strip().startswith(('#', '//', '/*', '*')):
                        code_lines.append(line)
                    if len(code_lines) > 50:  # Reasonable limit
                        break
                return '\n'.join(code_lines).strip()
        
        return text.strip()

# ============================================================================
# GEMINI MODEL IMPLEMENTATION (with Veo 3 & Imagen 4)
# ============================================================================

class GeminiModel(AIModelInterface):
    """Google Gemini model implementation with multimedia capabilities"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_config = config.get_model_config("gemini")
        self.session = None
        self.veo_enabled = self.model_config.get("enable_veo", False)
        self.imagen_enabled = self.model_config.get("enable_imagen", False)
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.model_config.get("timeout", 60))
            )
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code using Gemini with optional multimedia"""
        await self._ensure_session()
        
        start_time = datetime.now()
        
        try:
            # Prepare prompt for Gemini
            prompt = self._prepare_gemini_prompt(request)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Use API key in query parameter for Gemini
            url = f"{self.model_config['api_url']}/v1beta/models/{self.model_config['default_model']}:generateContent"
            url += f"?key={self.model_config['api_key']}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": request.temperature if request.temperature else self.model_config["temperature"],
                    "maxOutputTokens": min(request.max_tokens, self.model_config["max_tokens"]),
                    "topP": 0.95,
                    "topK": 40
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Extract generated text
                if "candidates" in result and result["candidates"]:
                    generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    generated_text = ""
                
                # Extract code
                code = self._extract_gemini_code(generated_text)
                
                # Generate teaching resources with potential multimedia
                teaching_resources = []
                if request.include_explanations:
                    teaching_resources = await self._generate_gemini_teaching_resources(
                        code, generated_text, request
                    )
                
                # Create response
                code_response = CodeGenerationResponse(
                    code=code,
                    metadata={
                        "model": self.model_config["default_model"],
                        "safety_ratings": result.get("promptFeedback", {}),
                        "citation_metadata": result.get("citationMetadata", {})
                    },
                    success=True,
                    model_used=self.model_config["default_model"],
                    provider=ModelProvider.GEMINI,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    token_usage={
                        "prompt_tokens": result.get("usageMetadata", {}).get("promptTokenCount", 0),
                        "candidates_tokens": result.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                        "total_tokens": result.get("usageMetadata", {}).get("totalTokenCount", 0)
                    },
                    confidence_score=self._calculate_gemini_confidence(result),
                    teaching_resources=teaching_resources,
                    raw_responses={"gemini": result}
                )
                
                return code_response
                
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used=self.model_config["default_model"],
                provider=ModelProvider.GEMINI,
                execution_time=(datetime.now() - start_time).total_seconds(),
                token_usage={},
                confidence_score=0.0,
                error=str(e)
            )
    
    def _prepare_gemini_prompt(self, request: CodeGenerationRequest) -> str:
        """Prepare prompt for Gemini model"""
        
        prompt_parts = []
        
        # Gemini-specific instructions
        prompt_parts.append("You are an expert programming assistant specializing in code generation.")
        
        # Task description
        prompt_parts.append(f"Generate high-quality {request.language.value} code for the following requirement:")
        prompt_parts.append(f"\"{request.prompt}\"")
        
        # Context
        if request.context:
            prompt_parts.append("\nAdditional context:")
            prompt_parts.append(request.context)
        
        # Requirements
        prompt_parts.append("\nRequirements:")
        prompt_parts.append(f"1. Language: {request.language.value}")
        prompt_parts.append("2. Include comprehensive comments explaining logic and key concepts")
        prompt_parts.append("3. Follow industry best practices and coding standards")
        prompt_parts.append("4. Ensure code is modular, readable, and maintainable")
        
        if request.include_tests:
            prompt_parts.append("5. Include unit tests with good coverage")
        
        if request.performance_optimize:
            prompt_parts.append("6. Optimize for performance and efficiency")
        
        if request.security_scan:
            prompt_parts.append("7. Implement security best practices")
        
        if request.file_name:
            prompt_parts.append(f"8. Appropriate for file: {request.file_name}")
        
        # Complexity handling
        if request.complexity == TaskComplexity.COMPLEX:
            prompt_parts.append("9. This is a complex task - provide sophisticated solution")
        elif request.complexity == TaskComplexity.EXPERT:
            prompt_parts.append("9. This is an expert-level task - provide advanced, optimized solution")
        
        # Custom instructions
        for key, value in request.custom_instructions.items():
            prompt_parts.append(f"{len(prompt_parts) - 8}. {key}: {value}")
        
        # Output format
        prompt_parts.append("\nProvide the complete code implementation within ``` code blocks.")
        if request.include_explanations:
            prompt_parts.append("Also include a brief explanation of the key concepts and design decisions.")
        
        return "\n".join(prompt_parts)
    
    def _extract_gemini_code(self, text: str) -> str:
        """Extract code from Gemini response"""
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        
        if code_blocks:
            # Join all code blocks
            return '\n\n'.join([block.strip() for block in code_blocks])
        
        # If no code blocks, return the entire text
        return text.strip()
    
    async def _generate_gemini_teaching_resources(self, code: str, explanation: str,
                                                request: CodeGenerationRequest) -> List[Dict]:
        """Generate teaching resources with Gemini's multimedia capabilities"""
        
        resources = []
        
        # Text explanation
        if explanation:
            resources.append({
                "type": "text_explanation",
                "title": "Code Explanation",
                "content": explanation,
                "format": "markdown",
                "difficulty": "intermediate"
            })
        
        # Generate diagram prompt for complex code
        if request.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT] and self.imagen_enabled:
            diagram_prompt = self._create_diagram_prompt(code, request.language)
            resources.append({
                "type": "diagram_suggestion",
                "title": "Suggested Architecture Diagram",
                "content": "Consider creating a diagram showing: " + diagram_prompt,
                "format": "text",
                "difficulty": "advanced"
            })
        
        # Extract key learning points
        learning_points = self._extract_learning_points(code, request.language)
        for point in learning_points:
            resources.append({
                "type": "learning_point",
                "title": point["title"],
                "content": point["explanation"],
                "format": "markdown",
                "difficulty": point["difficulty"],
                "code_snippet": point["example"]
            })
        
        return resources
    
    def _create_diagram_prompt(self, code: str, language: CodeLanguage) -> str:
        """Create prompt for generating architecture diagram"""
        
        # Extract key components for diagram
        components = []
        
        if language == CodeLanguage.PYTHON:
            # Find classes
            classes = re.findall(r'class\s+(\w+)', code)
            components.extend([f"Class: {cls}" for cls in classes])
            
            # Find main functions
            functions = re.findall(r'def\s+(\w+)\s*\(', code)
            if functions:
                components.append(f"Main functions: {', '.join(functions[:3])}")
        
        return f"Code structure showing {', '.join(components[:5])}"
    
    def _extract_learning_points(self, code: str, language: CodeLanguage) -> List[Dict]:
        """Extract learning points from code"""
        
        points = []
        
        if language == CodeLanguage.PYTHON:
            # Check for decorators
            if '@' in code:
                points.append({
                    "title": "Python Decorators",
                    "explanation": "Decorators modify or extend function behavior without changing their source code.",
                    "difficulty": "intermediate",
                    "example": "@decorator\ndef function(): pass"
                })
            
            # Check for context managers
            if 'with ' in code:
                points.append({
                    "title": "Context Managers",
                    "explanation": "Context managers handle resource setup and teardown automatically using 'with' statements.",
                    "difficulty": "intermediate",
                    "example": "with open('file.txt') as f:\n    content = f.read()"
                })
            
            # Check for type hints
            if '->' in code or ': List[' in code or ': Dict[' in code:
                points.append({
                    "title": "Type Hints",
                    "explanation": "Type hints improve code readability and enable static type checking.",
                    "difficulty": "intermediate",
                    "example": "def greet(name: str) -> str:\n    return f'Hello {name}'"
                })
        
        return points
    
    def _calculate_gemini_confidence(self, result: Dict) -> float:
        """Calculate confidence score for Gemini response"""
        
        score = 0.7  # Base score
        
        # Check safety ratings
        safety_feedback = result.get("promptFeedback", {})
        if safety_feedback and safety_feedback.get("blockReason"):
            score -= 0.3
        
        # Check token usage (more tokens often means more detailed)
        usage = result.get("usageMetadata", {})
        if usage.get("totalTokenCount", 0) > 100:
            score += 0.1
        
        # Check if response has candidates
        if result.get("candidates") and len(result["candidates"]) > 0:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    async def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze code using Gemini"""
        await self._ensure_session()
        
        try:
            prompt = f"""
            Perform comprehensive analysis of this {language.value} code:
            
            {code}
            
            Provide analysis in JSON format with:
            - quality_rating (1-10)
            - security_assessment
            - performance_analysis
            - readability_score (1-10)
            - maintainability_score (1-10)
            - identified_issues (list)
            - recommendations (list)
            - complexity_level (simple/medium/complex)
            """
            
            headers = {
                "Content-Type": "application/json"
            }
            
            url = f"{self.model_config['api_url']}/v1beta/models/{self.model_config['default_model']}:generateContent"
            url += f"?key={self.model_config['api_key']}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2000
                }
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "candidates" in result and result["candidates"]:
                    analysis_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    try:
                        # Try to parse as JSON
                        analysis = json.loads(analysis_text)
                    except json.JSONDecodeError:
                        # Fallback to text parsing
                        analysis = self._parse_gemini_analysis(analysis_text)
                    
                    return analysis
                else:
                    return {"error": "No analysis generated"}
                
        except Exception as e:
            logger.error(f"Gemini analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _parse_gemini_analysis(self, text: str) -> Dict[str, Any]:
        """Parse Gemini analysis text"""
        
        analysis = {
            "quality_rating": 5,
            "security_assessment": "Unknown",
            "performance_analysis": "Not analyzed",
            "readability_score": 5,
            "maintainability_score": 5,
            "identified_issues": [],
            "recommendations": [],
            "complexity_level": "medium"
        }
        
        # Simple parsing logic
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if "quality" in line_lower and any(str(i) in line for i in range(1, 11)):
                for i in range(1, 11):
                    if f"{i}/10" in line or f"rating: {i}" in line_lower:
                        analysis["quality_rating"] = i
                        break
            
            elif "security" in line_lower and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    analysis["security_assessment"] = parts[1].strip()
            
            elif "readability" in line_lower and any(str(i) in line for i in range(1, 11)):
                for i in range(1, 11):
                    if f"{i}/10" in line or f"score: {i}" in line_lower:
                        analysis["readability_score"] = i
                        break
            
            elif "issue" in line_lower and "- " in line:
                analysis["identified_issues"].append(line.strip())
            
            elif "recommend" in line_lower and "- " in line:
                analysis["recommendations"].append(line.strip())
        
        return analysis
    
    async def explain_code(self, code: str, language: CodeLanguage,
                          detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain code with Gemini"""
        await self._ensure_session()
        
        try:
            detail_instruction = {
                "simple": "Explain in simple, non-technical terms suitable for beginners",
                "detailed": "Provide detailed technical explanation with examples",
                "expert": "Provide expert-level analysis with optimizations and alternatives"
            }.get(detail_level, "Explain")
            
            prompt = f"""
            {detail_instruction} the following {language.value} code:
            
            {code}
            
            Focus on:
            1. What the code does
            2. How it works
            3. Key algorithms/patterns used
            4. Potential use cases
            5. Alternative approaches
            """
            
            headers = {
                "Content-Type": "application/json"
            }
            
            url = f"{self.model_config['api_url']}/v1beta/models/{self.model_config['default_model']}:generateContent"
            url += f"?key={self.model_config['api_key']}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 3000
                }
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "candidates" in result and result["candidates"]:
                    explanation = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    return {
                        "explanation": explanation,
                        "detail_level": detail_level,
                        "source": "Gemini",
                        "estimated_comprehension_time": f"{len(explanation.split()) // 150} minutes"
                    }
                else:
                    return {"explanation": "No explanation generated", "error": "Empty response"}
                
        except Exception as e:
            logger.error(f"Gemini explanation error: {str(e)}")
            return {"explanation": f"Error: {str(e)}", "detail_level": detail_level}
    
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str) -> Dict[str, Any]:
        """Refactor code with Gemini"""
        await self._ensure_session()
        
        try:
            prompt = f"""
            Refactor this {language.value} code with focus on {optimization_target}:
            
            {code}
            
            Provide:
            1. Refactored code with improvements
            2. Summary of changes made
            3. Benefits of the refactoring
            4. Before/after comparison
            5. Performance impact (if applicable)
            
            Optimization target: {optimization_target}
            """
            
            headers = {
                "Content-Type": "application/json"
            }
            
            url = f"{self.model_config['api_url']}/v1beta/models/{self.model_config['default_model']}:generateContent"
            url += f"?key={self.model_config['api_key']}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4000
                }
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "candidates" in result and result["candidates"]:
                    refactored_output = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Extract refactored code
                    refactored_code = self._extract_refactored_code(refactored_output)
                    
                    return {
                        "original_code": code,
                        "refactored_code": refactored_code,
                        "optimization_target": optimization_target,
                        "explanation": refactored_output,
                        "improvement_areas": self._identify_improvement_areas(code, refactored_code)
                    }
                else:
                    return {
                        "original_code": code,
                        "refactored_code": code,
                        "error": "No refactoring generated"
                    }
                
        except Exception as e:
            logger.error(f"Gemini refactoring error: {str(e)}")
            return {
                "original_code": code,
                "refactored_code": code,
                "error": str(e)
            }
    
    def _extract_refactored_code(self, text: str) -> str:
        """Extract refactored code from Gemini response"""
        return self._extract_gemini_code(text)  # Reuse same method
    
    def _identify_improvement_areas(self, original: str, refactored: str) -> List[str]:
        """Identify areas of improvement between original and refactored code"""
        
        improvements = []
        
        # Compare lengths (simplistic metric)
        if len(refactored) < len(original) * 0.8:
            improvements.append("Code length reduced significantly")
        
        # Check for added comments
        original_comments = len(re.findall(r'#.*|""".*?"''|//.*|/\*.*?\*/', original, re.DOTALL))
        refactored_comments = len(re.findall(r'#.*|""".*?"''|//.*|/\*.*?\*/', refactored, re.DOTALL))
        
        if refactored_comments > original_comments:
            improvements.append("Improved documentation/comments")
        
        # Check for function splitting
        original_funcs = len(re.findall(r'def\s+\w+\s*\(|function\s+\w+', original))
        refactored_funcs = len(re.findall(r'def\s+\w+\s*\(|function\s+\w+', refactored))
        
        if refactored_funcs > original_funcs:
            improvements.append("Better function modularity")
        
        return improvements if improvements else ["Code structure improved"]

# ============================================================================
# CLAUDE MODEL IMPLEMENTATION
# ============================================================================

class ClaudeModel(AIModelInterface):
    """Anthropic Claude model implementation with agentic capabilities"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_config = config.get_model_config("claude")
        self.session = None
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.model_config.get("timeout", 60))
            )
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code using Claude with agentic reasoning"""
        await self._ensure_session()
        
        start_time = datetime.now()
        
        try:
            # Prepare system prompt and user prompt for Claude
            system_prompt = self._prepare_claude_system_prompt(request)
            user_prompt = self._prepare_claude_user_prompt(request)
            
            headers = {
                "x-api-key": self.model_config["api_key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "max_tokens": min(request.max_tokens, self.model_config["max_tokens"]),
                "temperature": request.temperature if request.temperature else self.model_config["temperature"],
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            url = f"{self.model_config['api_url']}/v1/messages"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Extract content
                if "content" in result and result["content"]:
                    generated_text = result["content"][0]["text"]
                else:
                    generated_text = ""
                
                # Extract code
                code = self._extract_claude_code(generated_text)
                
                # Generate teaching resources
                teaching_resources = []
                if request.include_explanations:
                    teaching_resources = await self._generate_claude_teaching_resources(
                        code, generated_text, request
                    )
                
                # Create response
                code_response = CodeGenerationResponse(
                    code=code,
                    metadata={
                        "model": self.model_config["default_model"],
                        "stop_reason": result.get("stop_reason", "end_turn"),
                        "stop_sequence": result.get("stop_sequence")
                    },
                    success=True,
                    model_used=self.model_config["default_model"],
                    provider=ModelProvider.CLAUDE,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    token_usage={
                        "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": result.get("usage", {}).get("output_tokens", 0)
                    },
                    confidence_score=self._calculate_claude_confidence(result),
                    teaching_resources=teaching_resources,
                    raw_responses={"claude": result}
                )
                
                return code_response
                
        except Exception as e:
            logger.error(f"Claude generation error: {str(e)}")
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used=self.model_config["default_model"],
                provider=ModelProvider.CLAUDE,
                execution_time=(datetime.now() - start_time).total_seconds(),
                token_usage={},
                confidence_score=0.0,
                error=str(e)
            )
    
    def _prepare_claude_system_prompt(self, request: CodeGenerationRequest) -> str:
        """Prepare system prompt for Claude"""
        
        system_parts = []
        
        system_parts.append("You are Claude, an AI assistant specialized in code generation and software development.")
        system_parts.append("Your task is to generate high-quality, production-ready code based on user requirements.")
        
        # Language-specific expertise
        system_parts.append(f"You are particularly skilled in {request.language.value} programming.")
        
        # Quality requirements
        system_parts.append("Always follow these principles:")
        system_parts.append("1. Generate clean, readable, and maintainable code")
        system_parts.append("2. Include appropriate comments and documentation")
        system_parts.append("3. Follow language-specific best practices and conventions")
        system_parts.append("4. Consider edge cases and error handling")
        system_parts.append("5. Optimize for both performance and readability")
        
        if request.security_scan:
            system_parts.append("6. Implement security best practices and avoid vulnerabilities")
        
        if request.performance_optimize:
            system_parts.append("7. Optimize algorithms and data structures for performance")
        
        # Teaching aspect
        if request.include_explanations:
            system_parts.append("8. Provide explanations of key concepts and design decisions")
        
        return "\n".join(system_parts)
    
    def _prepare_claude_user_prompt(self, request: CodeGenerationRequest) -> str:
        """Prepare user prompt for Claude"""
        
        prompt_parts = []
        
        prompt_parts.append(f"Generate {request.language.value} code for the following requirement:")
        prompt_parts.append(f"{request.prompt}")
        
        if request.context:
            prompt_parts.append(f"\nContext:\n{request.context}")
        
        prompt_parts.append(f"\nSpecific requirements:")
        prompt_parts.append(f"- Language: {request.language.value}")
        
        if request.file_name:
            prompt_parts.append(f"- File name: {request.file_name}")
        
        if request.include_tests:
            prompt_parts.append("- Include comprehensive unit tests")
        
        if request.complexity == TaskComplexity.COMPLEX:
            prompt_parts.append("- This is a complex task requiring sophisticated solution")
        elif request.complexity == TaskComplexity.EXPERT:
            prompt_parts.append("- This is an expert-level task requiring advanced, optimized solution")
        
        # Custom instructions
        for key, value in request.custom_instructions.items():
            prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.append("\nPlease provide the complete code implementation.")
        
        if request.include_explanations:
            prompt_parts.append("Also include explanations of key concepts and design decisions.")
        
        return "\n".join(prompt_parts)
    
    def _extract_claude_code(self, text: str) -> str:
        """Extract code from Claude response"""
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        
        if code_blocks:
            # Join all code blocks
            return '\n\n'.join([block.strip() for block in code_blocks])
        
        # Claude often uses indented code blocks
        lines = text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for code block start
            if stripped.startswith(("def ", "class ", "function ", "const ", "let ", "var ", 
                                   "import ", "export ", "public ", "private ", "protected ")):
                in_code_block = True
            
            if in_code_block:
                # Check for end of code (blank line after indented code)
                if not stripped and code_lines and not code_lines[-1].strip():
                    in_code_block = False
                else:
                    code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return text.strip()
    
    async def _generate_claude_teaching_resources(self, code: str, explanation: str,
                                                request: CodeGenerationRequest) -> List[Dict]:
        """Generate teaching resources with Claude's explanatory style"""
        
        resources = []
        
        # Claude's explanations are usually comprehensive
        if explanation:
            # Split explanation from code
            explanation_text = explanation
            if '```' in explanation:
                # Extract text before first code block
                parts = explanation.split('```')
                if parts:
                    explanation_text = parts[0].strip()
            
            if explanation_text:
                resources.append({
                    "type": "comprehensive_explanation",
                    "title": "Detailed Code Explanation",
                    "content": explanation_text,
                    "format": "markdown",
                    "difficulty": "intermediate"
                })
        
        # Extract programming patterns
        patterns = self._identify_programming_patterns(code, request.language)
        for pattern in patterns:
            resources.append({
                "type": "programming_pattern",
                "title": f"Pattern: {pattern['name']}",
                "content": pattern['description'],
                "format": "markdown",
                "difficulty": pattern['level'],
                "example": pattern['example']
            })
        
        # Best practices analysis
        best_practices = self._analyze_best_practices(code, request.language)
        if best_practices:
            resources.append({
                "type": "best_practices",
                "title": "Applied Best Practices",
                "content": "\n".join(f"- {bp}" for bp in best_practices),
                "format": "markdown",
                "difficulty": "intermediate"
            })
        
        return resources
    
    def _identify_programming_patterns(self, code: str, language: CodeLanguage) -> List[Dict]:
        """Identify programming patterns in code"""
        
        patterns = []
        
        if language == CodeLanguage.PYTHON:
            # Check for decorator pattern
            if '@' in code:
                patterns.append({
                    "name": "Decorator Pattern",
                    "description": "A structural pattern that adds behavior to objects dynamically.",
                    "level": "intermediate",
                    "example": "@decorator\ndef function(): ..."
                })
            
            # Check for context manager pattern
            if 'with ' in code and 'open(' in code:
                patterns.append({
                    "name": "Context Manager Pattern",
                    "description": "Manages resources automatically using the 'with' statement.",
                    "level": "intermediate",
                    "example": "with open('file.txt') as f:\n    data = f.read()"
                })
            
            # Check for generator pattern
            if 'yield ' in code:
                patterns.append({
                    "name": "Generator Pattern",
                    "description": "Produces sequence of values lazily using yield.",
                    "level": "intermediate",
                    "example": "def generator():\n    yield 1\n    yield 2"
                })
        
        return patterns
    
    def _analyze_best_practices(self, code: str, language: CodeLanguage) -> List[str]:
        """Analyze code for applied best practices"""
        
        practices = []
        
        if language == CodeLanguage.PYTHON:
            # Check for docstrings
            if '"""' in code or "'''" in code:
                practices.append("Uses docstrings for documentation")
            
            # Check for type hints
            if '->' in code or ': ' in code and 'def ' in code:
                practices.append("Uses type hints for better code clarity")
            
            # Check for error handling
            if 'try:' in code and 'except:' in code:
                practices.append("Implements proper error handling")
            
            # Check for modular functions
            func_count = len(re.findall(r'def\s+\w+\s*\(', code))
            if func_count > 1:
                practices.append("Uses modular function design")
        
        return practices
    
    def _calculate_claude_confidence(self, result: Dict) -> float:
        """Calculate confidence score for Claude response"""
        
        score = 0.75  # Base score for Claude
        
        # Check stop reason
        stop_reason = result.get("stop_reason", "")
        if stop_reason == "end_turn":
            score += 0.1
        elif stop_reason == "max_tokens":
            score -= 0.1
        
        # Check token usage
        usage = result.get("usage", {})
        output_tokens = usage.get("output_tokens", 0)
        
        if output_tokens > 500:  # Substantial response
            score += 0.1
        elif output_tokens < 100:  # Very short response
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    async def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze code using Claude"""
        await self._ensure_session()
        
        try:
            system_prompt = f"""You are a code analysis expert. Analyze {language.value} code comprehensively."""
            
            user_prompt = f"""
            Analyze this {language.value} code comprehensively:
            
            {code}
            
            Provide analysis in this structured format:
            
            QUALITY_SCORE: X/10
            SECURITY_ASSESSMENT: [assessment]
            PERFORMANCE: [analysis]
            READABILITY: X/10
            MAINTAINABILITY: X/10
            
            ISSUES:
            - [issue 1]
            - [issue 2]
            
            RECOMMENDATIONS:
            - [recommendation 1]
            - [recommendation 2]
            
            COMPLEXITY: [simple/medium/complex]
            """
            
            headers = {
                "x-api-key": self.model_config["api_key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "max_tokens": 2000,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            
            url = f"{self.model_config['api_url']}/v1/messages"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "content" in result and result["content"]:
                    analysis_text = result["content"][0]["text"]
                    analysis = self._parse_claude_analysis(analysis_text)
                    return analysis
                else:
                    return {"error": "No analysis generated"}
                
        except Exception as e:
            logger.error(f"Claude analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _parse_claude_analysis(self, text: str) -> Dict[str, Any]:
        """Parse Claude's analysis response"""
        
        analysis = {
            "quality_score": 5,
            "security_assessment": "Unknown",
            "performance": "Not analyzed",
            "readability": 5,
            "maintainability": 5,
            "issues": [],
            "recommendations": [],
            "complexity": "medium"
        }
        
        # Parse structured response
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("QUALITY_SCORE:"):
                match = re.search(r'(\d+)/10', line)
                if match:
                    analysis["quality_score"] = int(match.group(1))
            
            elif line.startswith("SECURITY_ASSESSMENT:"):
                analysis["security_assessment"] = line.replace("SECURITY_ASSESSMENT:", "").strip()
            
            elif line.startswith("PERFORMANCE:"):
                analysis["performance"] = line.replace("PERFORMANCE:", "").strip()
            
            elif line.startswith("READABILITY:"):
                match = re.search(r'(\d+)/10', line)
                if match:
                    analysis["readability"] = int(match.group(1))
            
            elif line.startswith("MAINTAINABILITY:"):
                match = re.search(r'(\d+)/10', line)
                if match:
                    analysis["maintainability"] = int(match.group(1))
            
            elif line == "ISSUES:":
                current_section = "issues"
            elif line == "RECOMMENDATIONS:":
                current_section = "recommendations"
            elif line.startswith("COMPLEXITY:"):
                complexity = line.replace("COMPLEXITY:", "").strip().lower()
                if complexity in ["simple", "medium", "complex"]:
                    analysis["complexity"] = complexity
            
            elif current_section == "issues" and line.startswith("- "):
                analysis["issues"].append(line[2:].strip())
            elif current_section == "recommendations" and line.startswith("- "):
                analysis["recommendations"].append(line[2:].strip())
        
        return analysis
    
    async def explain_code(self, code: str, language: CodeLanguage,
                          detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain code with Claude"""
        await self._ensure_session()
        
        try:
            detail_map = {
                "simple": "Explain in simple, beginner-friendly terms",
                "detailed": "Provide detailed technical explanation",
                "expert": "Provide expert analysis with deep insights"
            }
            
            system_prompt = f"You are a programming educator explaining {language.value} code."
            
            user_prompt = f"""
            {detail_map.get(detail_level, "Explain")} this {language.value} code:
            
            {code}
            
            Focus on:
            1. Overall purpose and functionality
            2. Key algorithms and data structures
            3. Important functions/methods
            4. Flow and logic
            5. Potential use cases
            """
            
            headers = {
                "x-api-key": self.model_config["api_key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "max_tokens": 3000,
                "temperature": 0.4,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            
            url = f"{self.model_config['api_url']}/v1/messages"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "content" in result and result["content"]:
                    explanation = result["content"][0]["text"]
                    
                    return {
                        "explanation": explanation,
                        "detail_level": detail_level,
                        "estimated_learning_time": f"{len(explanation.split()) // 150} minutes",
                        "key_concepts": self._extract_key_concepts(explanation)
                    }
                else:
                    return {"explanation": "", "error": "No explanation generated"}
                
        except Exception as e:
            logger.error(f"Claude explanation error: {str(e)}")
            return {"explanation": f"Error: {str(e)}", "detail_level": detail_level}
    
    def _extract_key_concepts(self, explanation: str) -> List[str]:
        """Extract key concepts from explanation"""
        
        concepts = []
        
        # Look for emphasized terms (in bold, quotes, or followed by explanation)
        lines = explanation.split('\n')
        
        for line in lines:
            # Look for patterns like "Concept: explanation" or "**Concept** - explanation"
            if ':' in line and len(line.split(':')) > 1:
                concept = line.split(':')[0].strip()
                if concept and len(concept.split()) <= 3:  # Simple concept
                    concepts.append(concept)
            
            # Look for bold terms
            bold_matches = re.findall(r'\*\*(.*?)\*\*', line)
            concepts.extend(bold_matches)
        
        # Remove duplicates and limit
        return list(dict.fromkeys(concepts))[:5]
    
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str) -> Dict[str, Any]:
        """Refactor code with Claude"""
        await self._ensure_session()
        
        try:
            system_prompt = f"You are a code refactoring expert specializing in {language.value}."
            
            user_prompt = f"""
            Refactor this {language.value} code with focus on {optimization_target}:
            
            {code}
            
            Provide:
            1. The refactored code with improvements
            2. Summary of key changes made
            3. Benefits of each change
            4. Performance considerations
            5. Readability improvements
            
            Optimization target: {optimization_target}
            """
            
            headers = {
                "x-api-key": self.model_config["api_key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_config["default_model"],
                "max_tokens": 4000,
                "temperature": 0.3,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            
            url = f"{self.model_config['api_url']}/v1/messages"
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "content" in result and result["content"]:
                    refactored_output = result["content"][0]["text"]
                    
                    # Extract refactored code
                    refactored_code = self._extract_claude_code(refactored_output)
                    
                    return {
                        "original_code": code,
                        "refactored_code": refactored_code,
                        "optimization_target": optimization_target,
                        "change_summary": self._extract_change_summary(refactored_output),
                        "improvement_metrics": self._calculate_improvement_metrics(code, refactored_code)
                    }
                else:
                    return {
                        "original_code": code,
                        "refactored_code": code,
                        "error": "No refactoring generated"
                    }
                
        except Exception as e:
            logger.error(f"Claude refactoring error: {str(e)}")
            return {
                "original_code": code,
                "refactored_code": code,
                "error": str(e)
            }
    
    def _extract_change_summary(self, text: str) -> str:
        """Extract change summary from Claude response"""
        
        # Look for summary section
        lines = text.split('\n')
        summary_lines = []
        in_summary = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in 
                  ['summary', 'changes made', 'key changes']):
                in_summary = True
                continue
            
            if in_summary:
                if line.strip() and not line.startswith(('#', '//', '/*', '```')):
                    summary_lines.append(line)
                elif line.startswith('```'):
                    break
        
        if summary_lines:
            return '\n'.join(summary_lines).strip()
        
        # Fallback: first 200 characters
        return text[:200].strip() + "..."
    
    def _calculate_improvement_metrics(self, original: str, refactored: str) -> Dict[str, Any]:
        """Calculate improvement metrics between original and refactored code"""
        
        metrics = {
            "lines_changed": abs(len(refactored.split('\n')) - len(original.split('\n'))),
            "complexity_change": "unknown",
            "readability_improvement": "unknown"
        }
        
        # Simple complexity estimation
        orig_complexity = self._estimate_complexity(original)
        ref_complexity = self._estimate_complexity(refactored)
        
        if ref_complexity < orig_complexity:
            metrics["complexity_change"] = "reduced"
        elif ref_complexity > orig_complexity:
            metrics["complexity_change"] = "increased"
        else:
            metrics["complexity_change"] = "same"
        
        # Readability estimation (simplistic)
        orig_readability = self._estimate_readability(original)
        ref_readability = self._estimate_readability(refactored)
        
        if ref_readability > orig_readability:
            metrics["readability_improvement"] = "improved"
        elif ref_readability < orig_readability:
            metrics["readability_improvement"] = "worsened"
        else:
            metrics["readability_improvement"] = "same"
        
        return metrics
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity (simplistic)"""
        
        complexity = 0
        
        # Count control structures
        patterns = [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', 
                   r'\bexcept\b', r'\bcatch\b', r'\bswitch\b', r'\bcase\b']
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        # Count function/method definitions
        complexity += len(re.findall(r'def\s+\w+|function\s+\w+|class\s+\w+', code))
        
        # Count nested structures (simplistic)
        complexity += code.count('  ') // 10  # Rough indentation count
        
        return complexity
    
    def _estimate_readability(self, code: str) -> int:
        """Estimate code readability (simplistic)"""
        
        readability = 50  # Base score
        
        # Positive factors
        if '"""' in code or "'''" in code:  # Docstrings
            readability += 10
        
        if '# ' in code or '// ' in code:  # Comments
            readability += 5 * min(code.count('# ') + code.count('// '), 10)
        
        # Negative factors
        line_lengths = [len(line) for line in code.split('\n')]
        avg_line_length = sum(line_lengths) / max(len(line_lengths), 1)
        
        if avg_line_length > 80:  # Long lines
            readability -= 10
        
        # Check for very long functions (simplistic)
        lines = code.split('\n')
        in_function = False
        func_length = 0
        
        for line in lines:
            if re.match(r'\s*(def|function)\s+\w+', line):
                if in_function and func_length > 50:
                    readability -= 5
                in_function = True
                func_length = 0
            elif in_function:
                func_length += 1
        
        return max(min(readability, 100), 0)

# ============================================================================
# MODEL ROUTER AND ENSEMBLE SYSTEM
# ============================================================================

class ModelRouter:
    """Intelligent router for selecting and combining models"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.models = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all available models"""
        
        # Initialize DeepSeek R1
        if self.config.get_model_config("deepseek").get("enabled", False):
            self.models["deepseek"] = DeepSeekR1Model(self.config)
            self.performance_metrics["deepseek"] = ModelPerformanceMetrics(
                model_name="deepseek-r1",
                provider=ModelProvider.DEEPSEEK_R1
            )
        
        # Initialize Gemini
        if self.config.get_model_config("gemini").get("enabled", False):
            self.models["gemini"] = GeminiModel(self.config)
            self.performance_metrics["gemini"] = ModelPerformanceMetrics(
                model_name="gemini-2.5-pro",
                provider=ModelProvider.GEMINI
            )
        
        # Initialize Claude
        if self.config.get_model_config("claude").get("enabled", False):
            self.models["claude"] = ClaudeModel(self.config)
            self.performance_metrics["claude"] = ModelPerformanceMetrics(
                model_name="claude-3-5-sonnet",
                provider=ModelProvider.CLAUDE
            )
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    async def route_request(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Route request to appropriate model(s) based on intelligence"""
        
        # If specific provider requested, use it
        if request.model_provider:
            return await self._route_to_specific_provider(request)
        
        # Otherwise use intelligent routing
        best_model = self._select_best_model(request)
        
        if best_model == "ensemble":
            return await self._use_ensemble(request)
        else:
            return await self._route_to_model(best_model, request)
    
    def _select_best_model(self, request: CodeGenerationRequest) -> str:
        """Select the best model for the given request"""
        
        # Model selection logic based on request characteristics
        model_scores = {}
        
        for model_name in self.models.keys():
            score = self._calculate_model_score(model_name, request)
            model_scores[model_name] = score
        
        # Check if ensemble would be better
        ensemble_score = self._calculate_ensemble_score(request)
        model_scores["ensemble"] = ensemble_score
        
        # Select best model
        best_model = max(model_scores, key=model_scores.get)
        
        logger.info(f"Selected model: {best_model} with score: {model_scores[best_model]:.2f}")
        return best_model
    
    def _calculate_model_score(self, model_name: str, request: CodeGenerationRequest) -> float:
        """Calculate score for a model based on request characteristics"""
        
        score = 0.5  # Base score
        
        # Model-specific strengths
        if model_name == "deepseek":
            # DeepSeek R1 excels at reasoning and complex tasks
            if request.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
                score += 0.3
            if request.language == CodeLanguage.PYTHON:
                score += 0.1
        
        elif model_name == "gemini":
            # Gemini good for multimedia and latest features
            if request.include_explanations:
                score += 0.2
            if request.language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT, CodeLanguage.GO]:
                score += 0.1
        
        elif model_name == "claude":
            # Claude good for detailed explanations and safety
            if request.include_explanations:
                score += 0.3
            if request.security_scan:
                score += 0.2
        
        # Performance history
        metrics = self.performance_metrics.get(model_name)
        if metrics and metrics.call_count > 0:
            success_rate_factor = metrics.success_rate * 0.3
            confidence_factor = metrics.avg_confidence * 0.2
            score += success_rate_factor + confidence_factor
        
        # Adjust for recent performance
        if metrics and metrics.error_count > metrics.call_count * 0.1:  # >10% error rate
            score -= 0.2
        
        return max(min(score, 1.0), 0.0)
    
    def _calculate_ensemble_score(self, request: CodeGenerationRequest) -> float:
        """Calculate score for using ensemble approach"""
        
        base_score = 0.6  # Ensemble base score
        
        # Ensemble is good for:
        # 1. Complex tasks
        if request.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            base_score += 0.2
        
        # 2. When high accuracy is critical
        if request.performance_optimize or request.security_scan:
            base_score += 0.1
        
        # 3. When we have multiple models available
        available_models = len(self.models)
        if available_models >= 2:
            base_score += 0.1 * (available_models - 1)
        
        return min(base_score, 1.0)
    
    async def _route_to_specific_provider(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Route to a specific provider as requested"""
        
        provider_map = {
            ModelProvider.DEEPSEEK_R1: "deepseek",
            ModelProvider.GEMINI: "gemini",
            ModelProvider.CLAUDE: "claude",
            ModelProvider.ENSEMBLE: "ensemble"
        }
        
        model_key = provider_map.get(request.model_provider)
        
        if model_key == "ensemble":
            return await self._use_ensemble(request)
        elif model_key in self.models:
            return await self._route_to_model(model_key, request)
        else:
            # Fallback to default
            default_model = list(self.models.keys())[0] if self.models else None
            if default_model:
                return await self._route_to_model(default_model, request)
            else:
                return CodeGenerationResponse(
                    code="",
                    metadata={},
                    success=False,
                    model_used="none",
                    provider=ModelProvider.CUSTOM,
                    execution_time=0.0,
                    token_usage={},
                    confidence_score=0.0,
                    error="No models available"
                )
    
    async def _route_to_model(self, model_key: str, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Route request to specific model"""
        
        model = self.models.get(model_key)
        if not model:
            raise ValueError(f"Model {model_key} not found")
        
        start_time = datetime.now()
        
        try:
            response = await model.generate_code(request)
            
            # Update performance metrics
            self._update_performance_metrics(model_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Model {model_key} failed: {str(e)}")
            
            # Update error metrics
            if model_key in self.performance_metrics:
                self.performance_metrics[model_key].error_count += 1
            
            # Try fallback model
            return await self._try_fallback_model(model_key, request, str(e))
    
    async def _use_ensemble(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Use ensemble of models for best results"""
        
        if len(self.models) < 2:
            # Not enough models for ensemble, use best available
            best_model = self._select_best_model(request)
            if best_model == "ensemble":
                best_model = list(self.models.keys())[0]
            return await self._route_to_model(best_model, request)
        
        start_time = datetime.now()
        
        try:
            # Generate with all available models in parallel
            tasks = []
            for model_key in self.models.keys():
                task = asyncio.create_task(
                    self._route_to_model(model_key, request)
                )
                tasks.append((model_key, task))
            
            # Wait for all responses
            responses = []
            for model_key, task in tasks:
                try:
                    response = await asyncio.wait_for(task, timeout=60)
                    responses.append((model_key, response))
                except asyncio.TimeoutError:
                    logger.warning(f"Model {model_key} timed out")
                except Exception as e:
                    logger.error(f"Model {model_key} failed: {str(e)}")
            
            if not responses:
                raise Exception("All models failed")
            
            # Combine results
            combined_response = self._combine_responses(responses, request)
            
            # Update ensemble wasn't directly called, but update individual models
            for model_key, response in responses:
                if response.success:
                    self._update_performance_metrics(model_key, response)
            
            return combined_response
            
        except Exception as e:
            logger.error(f"Ensemble failed: {str(e)}")
            
            # Try single model as fallback
            best_model = list(self.models.keys())[0]
            return await self._route_to_model(best_model, request)
    
    def _combine_responses(self, responses: List[Tuple[str, CodeGenerationResponse]],
                          request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Combine multiple model responses intelligently"""
        
        # Filter successful responses
        successful = [(key, resp) for key, resp in responses if resp.success]
        
        if not successful:
            # No successful responses
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used="ensemble",
                provider=ModelProvider.ENSEMBLE,
                execution_time=0.0,
                token_usage={},
                confidence_score=0.0,
                error="All models failed"
            )
        
        if len(successful) == 1:
            # Only one successful response
            model_key, response = successful[0]
            response.model_used = f"ensemble-{model_key}"
            response.provider = ModelProvider.ENSEMBLE
            return response
        
        # Multiple successful responses - combine them
        # Strategy: Use highest confidence response as base, 
        # supplement with teaching resources from others
        
        # Sort by confidence
        successful.sort(key=lambda x: x[1].confidence_score, reverse=True)
        
        best_model_key, best_response = successful[0]
        
        # Combine teaching resources
        all_teaching_resources = []
        for _, response in successful:
            all_teaching_resources.extend(response.teaching_resources)
        
        # Remove duplicates (based on title)
        seen_titles = set()
        unique_resources = []
        for resource in all_teaching_resources:
            title = resource.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_resources.append(resource)
        
        # Combine warnings and suggestions
        all_warnings = set()
        all_suggestions = set()
        for _, response in successful:
            all_warnings.update(response.warnings)
            all_suggestions.update(response.suggestions)
        
        # Combine metadata
        combined_metadata = {
            "ensemble_models": [key for key, _ in successful],
            "confidence_scores": {key: resp.confidence_score for key, resp in successful},
            "base_model": best_model_key
        }
        
        # Update best response with combined info
        best_response.model_used = f"ensemble-{'-'.join([key for key, _ in successful])}"
        best_response.provider = ModelProvider.ENSEMBLE
        best_response.teaching_resources = unique_resources[:10]  # Limit to 10 resources
        best_response.warnings = list(all_warnings)
        best_response.suggestions = list(all_suggestions)
        best_response.metadata.update(combined_metadata)
        
        # Add note about ensemble
        if request.include_explanations and best_response.code:
            ensemble_note = "\n\n# Note: This code was generated using an ensemble of AI models "
            ensemble_note += f"({', '.join([key for key, _ in successful])}) "
            ensemble_note += "to ensure the best possible quality and accuracy."
            best_response.code += ensemble_note
        
        return best_response
    
    async def _try_fallback_model(self, failed_model: str, 
                                 request: CodeGenerationRequest, 
                                 error_msg: str) -> CodeGenerationResponse:
        """Try fallback model when primary fails"""
        
        available_models = [m for m in self.models.keys() if m != failed_model]
        
        if not available_models:
            # No fallback available
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used=failed_model,
                provider=ModelProvider.CUSTOM,
                execution_time=0.0,
                token_usage={},
                confidence_score=0.0,
                error=f"Primary model failed: {error_msg}"
            )
        
        # Try next best model
        fallback_model = available_models[0]
        logger.info(f"Falling back from {failed_model} to {fallback_model}")
        
        try:
            return await self._route_to_model(fallback_model, request)
        except Exception as e:
            logger.error(f"Fallback model also failed: {str(e)}")
            
            return CodeGenerationResponse(
                code="",
                metadata={},
                success=False,
                model_used=f"{failed_model},{fallback_model}",
                provider=ModelProvider.CUSTOM,
                execution_time=0.0,
                token_usage={},
                confidence_score=0.0,
                error=f"All models failed: {error_msg}, {str(e)}"
            )
    
    def _update_performance_metrics(self, model_key: str, response: CodeGenerationResponse):
        """Update performance metrics for a model"""
        
        if model_key not in self.performance_metrics:
            self.performance_metrics[model_key] = ModelPerformanceMetrics(
                model_name=model_key,
                provider=response.provider
            )
        
        metrics = self.performance_metrics[model_key]
        
        # Update metrics
        metrics.call_count += 1
        
        if response.success:
            metrics.success_rate = ((metrics.success_rate * (metrics.call_count - 1)) + 1) / metrics.call_count
            metrics.avg_confidence = ((metrics.avg_confidence * (metrics.call_count - 1)) + 
                                     response.confidence_score) / metrics.call_count
            
            # Update token efficiency (higher is better)
            tokens_used = response.token_usage.get("total_tokens", 0) or 0
            code_length = len(response.code)
            if tokens_used > 0:
                efficiency = code_length / tokens_used
                metrics.token_efficiency = ((metrics.token_efficiency * (metrics.call_count - 1)) + 
                                          efficiency) / metrics.call_count
        else:
            metrics.error_count += 1
            metrics.success_rate = (metrics.success_rate * (metrics.call_count - 1)) / metrics.call_count
        
        metrics.last_used = datetime.now()
    
    async def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze code using best available model"""
        
        # Use Claude for analysis if available (good at explanations)
        if "claude" in self.models:
            return await self.models["claude"].analyze_code(code, language)
        
        # Fallback to other models
        for model_key in ["deepseek", "gemini"]:
            if model_key in self.models:
                return await self.models[model_key].analyze_code(code, language)
        
        return {"error": "No models available for analysis"}
    
    async def explain_code(self, code: str, language: CodeLanguage,
                          detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain code using best available model"""
        
        # Use Claude for explanations if available
        if "claude" in self.models:
            return await self.models["claude"].explain_code(code, language, detail_level)
        
        # Fallback to other models
        for model_key in ["deepseek", "gemini"]:
            if model_key in self.models:
                return await self.models[model_key].explain_code(code, language, detail_level)
        
        return {"explanation": "No models available", "error": "Service unavailable"}
    
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str) -> Dict[str, Any]:
        """Refactor code using best available model"""
        
        # Use DeepSeek for refactoring if available (good at reasoning)
        if "deepseek" in self.models:
            return await self.models["deepseek"].refactor_code(code, language, optimization_target)
        
        # Fallback to other models
        for model_key in ["claude", "gemini"]:
            if model_key in self.models:
                return await self.models[model_key].refactor_code(code, language, optimization_target)
        
        return {"error": "No models available for refactoring"}

# ============================================================================
# SECURITY SCANNER IMPLEMENTATION
# ============================================================================

class SecurityScanner(SecurityScannerInterface):
    """Advanced security scanner for code analysis"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        
    def _load_vulnerability_patterns(self) -> Dict[str, List[Dict]]:
        """Load vulnerability patterns for different languages"""
        
        patterns = {
            "python": [
                {
                    "pattern": r'eval\s*\(',
                    "description": "Use of eval() function",
                    "severity": "high",
                    "mitigation": "Use safer alternatives like ast.literal_eval() or explicit parsing"
                },
                {
                    "pattern": r'exec\s*\(',
                    "description": "Use of exec() function",
                    "severity": "high",
                    "mitigation": "Avoid exec() with user input; use safer execution methods"
                },
                {
                    "pattern": r'pickle\.loads|pickle\.load',
                    "description": "Unsafe deserialization with pickle",
                    "severity": "critical",
                    "mitigation": "Use safer serialization formats like JSON or implement proper validation"
                },
                {
                    "pattern": r'subprocess\.(call|check_output|Popen)',
                    "description": "Shell command execution",
                    "severity": "medium",
                    "mitigation": "Validate and sanitize all inputs; use explicit command arguments"
                },
                {
                    "pattern": r'os\.system|os\.popen',
                    "description": "Direct OS command execution",
                    "severity": "high",
                    "mitigation": "Use subprocess with explicit arguments and input validation"
                },
                {
                    "pattern": r'sql\s*\+',
                    "description": "String concatenation in SQL queries",
                    "severity": "critical",
                    "mitigation": "Use parameterized queries or ORM"
                },
                {
                    "pattern": r'\.format\s*\(\s*[^\)]*\{.*?\}',
                    "description": "Potential string format vulnerability",
                    "severity": "medium",
                    "mitigation": "Validate format strings or use f-strings with care"
                }
            ],
            "javascript": [
                {
                    "pattern": r'eval\s*\(',
                    "description": "Use of eval() function",
                    "severity": "high",
                    "mitigation": "Avoid eval(); use JSON.parse() or Function() with caution"
                },
                {
                    "pattern": r'innerHTML\s*=',
                    "description": "Direct innerHTML assignment",
                    "severity": "high",
                    "mitigation": "Use textContent or DOM manipulation methods"
                },
                {
                    "pattern": r'document\.write\s*\(',
                    "description": "Direct document.write usage",
                    "severity": "medium",
                    "mitigation": "Use DOM manipulation methods instead"
                },
                {
                    "pattern": r'setTimeout\s*\(\s*[^,)]+\s*\)',
                    "description": "Dynamic code in setTimeout",
                    "severity": "medium",
                    "mitigation": "Pass function references instead of strings"
                },
                {
                    "pattern": r'localStorage|sessionStorage',
                    "description": "Client-side storage of sensitive data",
                    "severity": "medium",
                    "mitigation": "Avoid storing sensitive data client-side; use secure server storage"
                }
            ]
        }
        
        return patterns
    
    async def scan_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        
        start_time = datetime.now()
        
        try:
            vulnerabilities = []
            language_patterns = self.vulnerability_patterns.get(language.value, [])
            
            for pattern_info in language_patterns:
                matches = re.finditer(pattern_info["pattern"], code, re.IGNORECASE)
                
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    line_content = self._get_line_content(code, match.start())
                    
                    vulnerabilities.append({
                        "type": pattern_info["description"],
                        "severity": pattern_info["severity"],
                        "line": line_num,
                        "code_snippet": line_content[:100],
                        "mitigation": pattern_info["mitigation"],
                        "pattern": pattern_info["pattern"]
                    })
            
            # Additional security checks
            security_score = self._calculate_security_score(vulnerabilities, code)
            
            scan_result = {
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "security_score": security_score,
                "scan_time": (datetime.now() - start_time).total_seconds(),
                "language": language.value,
                "code_length": len(code),
                "recommendations": self._generate_security_recommendations(vulnerabilities, language)
            }
            
            return scan_result
            
        except Exception as e:
            logger.error(f"Security scan error: {str(e)}")
            return {
                "error": str(e),
                "vulnerabilities_found": 0,
                "security_score": 0,
                "scan_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_line_content(self, code: str, position: int) -> str:
        """Get the line content at given position"""
        
        lines = code.split('\n')
        current_pos = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_pos <= position < current_pos + line_length:
                return line
            current_pos += line_length
        
        return ""
    
    def _calculate_security_score(self, vulnerabilities: List[Dict], code: str) -> int:
        """Calculate security score (0-100)"""
        
        if not code.strip():
            return 0
        
        base_score = 100
        severity_weights = {
            "critical": 20,
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "medium")
            weight = severity_weights.get(severity, 5)
            base_score -= weight
        
        # Adjust based on code complexity
        lines = code.split('\n')
        if len(lines) > 0:
            # More complex code has more potential issues
            complexity_factor = min(len(lines) / 100, 2.0)  # Cap at 2x
            base_score = int(base_score / complexity_factor)
        
        return max(min(base_score, 100), 0)
    
    def _generate_security_recommendations(self, vulnerabilities: List[Dict], 
                                         language: CodeLanguage) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        if not vulnerabilities:
            recommendations.append("No major security issues detected.")
            recommendations.append("Always validate and sanitize user inputs.")
            recommendations.append("Keep dependencies updated.")
            return recommendations
        
        # Count vulnerabilities by severity
        severity_counts = {}
        for vuln in vulnerabilities:
            severity = vuln["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Add severity-based recommendations
        if severity_counts.get("critical", 0) > 0:
            recommendations.append(f"Fix {severity_counts['critical']} critical vulnerabilities immediately.")
        
        if severity_counts.get("high", 0) > 0:
            recommendations.append(f"Address {severity_counts['high']} high-severity vulnerabilities.")
        
        # Language-specific recommendations
        if language == CodeLanguage.PYTHON:
            recommendations.append("Use python's built-in security features like hashlib for hashing.")
            recommendations.append("Consider using security-focused libraries like bandit for static analysis.")
        
        elif language == CodeLanguage.JAVASCRIPT:
            recommendations.append("Implement Content Security Policy (CSP) headers.")
            recommendations.append("Use HTTPS for all connections.")
        
        # General recommendations
        recommendations.append("Implement input validation and output encoding.")
        recommendations.append("Use prepared statements for database queries.")
        recommendations.append("Regularly update dependencies and scan for vulnerabilities.")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def scan_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Scan dependencies for vulnerabilities"""
        
        # This would integrate with real dependency scanners
        # For now, return mock results
        
        mock_vulnerabilities = []
        
        for dep in dependencies[:10]:  # Limit scanning
            # Mock vulnerability detection
            if any(keyword in dep.lower() for keyword in ["old", "beta", "alpha", "rc"]):
                mock_vulnerabilities.append({
                    "dependency": dep,
                    "version": "unknown",
                    "severity": "medium",
                    "description": f"Potential outdated or unstable dependency: {dep}",
                    "recommendation": "Update to latest stable version"
                })
        
        return {
            "dependencies_scanned": len(dependencies),
            "vulnerabilities_found": len(mock_vulnerabilities),
            "vulnerabilities": mock_vulnerabilities,
            "recommendations": [
                "Regularly update all dependencies",
                "Use dependency vulnerability scanners like Dependabot or Snyk",
                "Pin dependency versions in requirements files"
            ]
        }

# ============================================================================
# PERFORMANCE OPTIMIZER IMPLEMENTATION
# ============================================================================

class PerformanceOptimizer(PerformanceOptimizerInterface):
    """Performance optimizer for code analysis and improvement"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        
    async def optimize_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Optimize code for performance"""
        
        start_time = datetime.now()
        
        try:
            optimization_suggestions = []
            language_optimizations = self._get_language_optimizations(language)
            
            # Apply language-specific optimizations
            for opt in language_optimizations:
                if self._should_apply_optimization(code, opt):
                    optimization_suggestions.append(opt)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(code, optimization_suggestions, language)
            
            # Generate optimized code (simplified - in practice would actually rewrite)
            optimized_code = self._apply_optimizations(code, optimization_suggestions, language)
            
            result = {
                "original_code_length": len(code),
                "optimized_code_length": len(optimized_code),
                "optimization_suggestions": optimization_suggestions,
                "optimization_score": optimization_score,
                "estimated_performance_improvement": self._estimate_improvement(optimization_suggestions),
                "optimized_code": optimized_code,
                "optimization_time": (datetime.now() - start_time).total_seconds(),
                "language": language.value
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Performance optimization error: {str(e)}")
            return {
                "error": str(e),
                "original_code": code,
                "optimized_code": code,
                "optimization_score": 0,
                "optimization_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_language_optimizations(self, language: CodeLanguage) -> List[Dict]:
        """Get language-specific optimization patterns"""
        
        optimizations = {
            "python": [
                {
                    "pattern": r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(\s*\w+\s*\)\s*\)',
                    "description": "Use enumerate() instead of range(len()) for index and value",
                    "improvement": "Readability and slight performance improvement",
                    "example_before": "for i in range(len(items)):\n    item = items[i]",
                    "example_after": "for i, item in enumerate(items):"
                },
                {
                    "pattern": r'\.join\s*\(\s*\[\s*str\s*\(\s*\w+\s*\)\s+for\s+\w+\s+in\s+\w+\s*\]\s*\)',
                    "description": "Use generator expression instead of list comprehension in join()",
                    "improvement": "Memory efficiency",
                    "example_before": "','.join([str(x) for x in items])",
                    "example_after": "','.join(str(x) for x in items)"
                },
                {
                    "pattern": r'if\s+\w+\s+in\s+\[\s*[^\]]+\s*\]',
                    "description": "Use set for membership testing with multiple values",
                    "improvement": "Faster lookup (O(1) vs O(n))",
                    "example_before": "if x in [1, 2, 3, 4, 5]:",
                    "example_after": "if x in {1, 2, 3, 4, 5}:"
                }
            ],
            "javascript": [
                {
                    "pattern": r'for\s*\(\s*var\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*\w+\.length\s*;\s*\w+\+\+\s*\)',
                    "description": "Cache array length in for loops",
                    "improvement": "Performance improvement for large arrays",
                    "example_before": "for (var i = 0; i < arr.length; i++)",
                    "example_after": "for (var i = 0, len = arr.length; i < len; i++)"
                },
                {
                    "pattern": r'console\.log\s*\(',
                    "description": "Remove console.log statements in production code",
                    "improvement": "Performance and security",
                    "example_before": "console.log('Debug:', value);",
                    "example_after": "// Remove or use proper logging library"
                }
            ]
        }
        
        return optimizations.get(language.value, [])
    
    def _should_apply_optimization(self, code: str, optimization: Dict) -> bool:
        """Check if optimization should be applied"""
        
        pattern = optimization.get("pattern", "")
        if not pattern:
            return False
        
        # Check if pattern exists in code
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _calculate_optimization_score(self, code: str, 
                                     optimizations: List[Dict], 
                                     language: CodeLanguage) -> int:
        """Calculate optimization score (0-100)"""
        
        if not code.strip():
            return 0
        
        base_score = 50  # Average code
        
        # Add points for each applicable optimization
        score_increase = min(len(optimizations) * 5, 30)  # Max 30 points from optimizations
        
        # Check for known anti-patterns
        anti_patterns = self._get_anti_patterns(language)
        anti_pattern_count = 0
        
        for pattern in anti_patterns:
            if re.search(pattern["pattern"], code, re.IGNORECASE):
                anti_pattern_count += 1
        
        score_decrease = min(anti_pattern_count * 10, 40)  # Max 40 points decrease
        
        final_score = base_score + score_increase - score_decrease
        
        return max(min(final_score, 100), 0)
    
    def _get_anti_patterns(self, language: CodeLanguage) -> List[Dict]:
        """Get language-specific anti-patterns"""
        
        anti_patterns = {
            "python": [
                {
                    "pattern": r'try:\s*[^\n]+\n\s*except:\s*pass',
                    "description": "Bare except clause that passes",
                    "severity": "high"
                },
                {
                    "pattern": r'from\s+\w+\s+import\s*\*',
                    "description": "Wildcard import",
                    "severity": "medium"
                }
            ],
            "javascript": [
                {
                    "pattern": r'with\s*\(',
                    "description": "with statement usage",
                    "severity": "high"
                },
                {
                    "pattern": r'==\s*null|\!=\s*null',
                    "description": "Loose equality with null",
                    "severity": "medium"
                }
            ]
        }
        
        return anti_patterns.get(language.value, [])
    
    def _estimate_improvement(self, optimizations: List[Dict]) -> str:
        """Estimate performance improvement from optimizations"""
        
        if not optimizations:
            return "Minimal (code already optimized)"
        
        improvement_levels = {
            1: "Slight (minor optimizations)",
            2: "Moderate (noticeable improvements)",
            3: "Significant (major performance gains)",
            4: "Substantial (dramatic improvement)",
            5: "Transformative (complete optimization)"
        }
        
        level = min(len(optimizations), 5)
        return improvement_levels.get(level, "Moderate improvements")
    
    def _apply_optimizations(self, code: str, optimizations: List[Dict], 
                            language: CodeLanguage) -> str:
        """Apply optimizations to code (simplified implementation)"""
        
        if not optimizations:
            return code
        
        optimized_code = code
        
        # For demonstration, we'll just add comments about optimizations
        # In a real implementation, this would actually rewrite the code
        
        optimization_comments = []
        for opt in optimizations[:3]:  # Limit to 3 optimizations for demo
            optimization_comments.append(f"# Optimization: {opt['description']}")
            optimization_comments.append(f"# Improvement: {opt['improvement']}")
        
        if optimization_comments:
            header = "\n".join(optimization_comments)
            optimized_code = f"{header}\n\n{optimized_code}"
        
        return optimized_code
    
    async def benchmark_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Benchmark code performance (simplified)"""
        
        start_time = datetime.now()
        
        try:
            # Simplified benchmarking
            # In reality, this would execute the code and measure performance
            
            lines = code.split('\n')
            functions = len(re.findall(r'def\s+\w+|function\s+\w+', code))
            loops = len(re.findall(r'\b(for|while)\b', code, re.IGNORECASE))
            
            # Estimate complexity
            complexity_score = self._estimate_complexity_score(code, language)
            
            # Performance metrics (simulated)
            estimated_runtime = self._estimate_runtime(code, language)
            memory_usage = self._estimate_memory_usage(code, language)
            
            benchmark_result = {
                "code_metrics": {
                    "lines_of_code": len(lines),
                    "functions": functions,
                    "loops": loops,
                    "complexity_score": complexity_score
                },
                "performance_estimates": {
                    "estimated_runtime_ms": estimated_runtime,
                    "estimated_memory_mb": memory_usage,
                    "cpu_intensity": self._estimate_cpu_intensity(code, language)
                },
                "bottlenecks": self._identify_bottlenecks(code, language),
                "benchmark_time": (datetime.now() - start_time).total_seconds(),
                "language": language.value
            }
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            return {
                "error": str(e),
                "benchmark_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _estimate_complexity_score(self, code: str, language: CodeLanguage) -> int:
        """Estimate code complexity score (0-100)"""
        
        score = 30  # Base
        
        # Add for various complexity factors
        score += min(len(code.split('\n')) // 10, 20)  # Lines of code
        score += min(len(re.findall(r'def\s+\w+|function\s+\w+', code)) * 5, 20)  # Functions
        score += min(len(re.findall(r'\b(if|else|elif|case|switch)\b', code, re.IGNORECASE)) * 3, 15)  # Conditionals
        score += min(len(re.findall(r'\b(for|while)\b', code, re.IGNORECASE)) * 5, 15)  # Loops
        
        return min(score, 100)
    
    def _estimate_runtime(self, code: str, language: CodeLanguage) -> float:
        """Estimate runtime in milliseconds (simplified)"""
        
        # Very simplified estimation
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+|function\s+\w+', code))
        loops = len(re.findall(r'\b(for|while)\b', code, re.IGNORECASE))
        
        # Base runtime estimation
        runtime = lines * 0.1 + functions * 0.5 + loops * 2.0
        
        return round(runtime, 2)
    
    def _estimate_memory_usage(self, code: str, language: CodeLanguage) -> float:
        """Estimate memory usage in MB (simplified)"""
        
        # Very simplified estimation
        lines = len(code.split('\n'))
        arrays = len(re.findall(r'\[\s*\]|\b(list|dict|set|array)\s*\(', code, re.IGNORECASE))
        
        memory = lines * 0.01 + arrays * 0.1
        
        return round(memory, 2)
    
    def _estimate_cpu_intensity(self, code: str, language: CodeLanguage) -> str:
        """Estimate CPU intensity"""
        
        loops = len(re.findall(r'\b(for|while)\b', code, re.IGNORECASE))
        recursions = len(re.findall(r'\bdef\s+\w+.*:\s*.*\1\(|function\s+\w+.*{.*\1\(', code))
        
        if recursions > 0 or loops > 10:
            return "High"
        elif loops > 5:
            return "Medium"
        else:
            return "Low"
    
    def _identify_bottlenecks(self, code: str, language: CodeLanguage) -> List[str]:
        """Identify potential performance bottlenecks"""
        
        bottlenecks = []
        
        # Check for nested loops
        nested_loops = re.findall(r'\bfor\b.*:\s*\n.*\bfor\b|\bwhile\b.*:\s*\n.*\bwhile\b', code, re.IGNORECASE)
        if nested_loops:
            bottlenecks.append("Nested loops detected - potential O(n²) complexity")
        
        # Check for recursion without base case
        if language == CodeLanguage.PYTHON:
            recursions = re.findall(r'def\s+(\w+).*:\s*.*\1\(', code)
            if recursions and not re.search(r'if\s+.*:\s*.*return', code, re.IGNORECASE):
                bottlenecks.append("Recursion without clear base case")
        
        # Check for large data structures in loops
        if re.search(r'for.*in.*:\s*.*\.append\(|while.*:\s*.*\.append\(', code, re.IGNORECASE):
            bottlenecks.append("Appending to lists/arrays inside loops")
        
        if not bottlenecks:
            bottlenecks.append("No obvious bottlenecks detected")
        
        return bottlenecks[:3]  # Limit to 3 bottlenecks

# ============================================================================
# TEACHING RESOURCE GENERATOR
# ============================================================================

class TeachingResourceGeneratorImpl(TeachingResourceGenerator):
    """Implementation of teaching resource generator"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        
    async def generate_explanation(self, concept: str, code: str, 
                                  language: CodeLanguage) -> Dict[str, Any]:
        """Generate concept explanation with examples"""
        
        try:
            # This would integrate with AI models for better explanations
            # For now, generate structured explanations
            
            explanation = self._create_concept_explanation(concept, code, language)
            examples = self._generate_examples(concept, language)
            common_mistakes = self._identify_common_mistakes(concept, language)
            
            return {
                "concept": concept,
                "explanation": explanation,
                "examples": examples,
                "common_mistakes": common_mistakes,
                "language": language.value,
                "difficulty_level": self._assess_difficulty(concept, language),
                "estimated_learning_time": "10-15 minutes",
                "prerequisites": self._identify_prerequisites(concept, language)
            }
            
        except Exception as e:
            logger.error(f"Explanation generation error: {str(e)}")
            return {
                "concept": concept,
                "error": str(e),
                "language": language.value
            }
    
    def _create_concept_explanation(self, concept: str, code: str, 
                                   language: CodeLanguage) -> str:
        """Create concept explanation"""
        
        explanations = {
            "python": {
                "function": "A function is a reusable block of code that performs a specific task. "
                          "Functions help organize code, avoid repetition, and make programs easier to understand.",
                "class": "A class is a blueprint for creating objects. It defines attributes (data) and "
                        "methods (functions) that the objects will have.",
                "loop": "Loops allow you to execute a block of code repeatedly. Python has 'for' loops "
                       "for iterating over sequences and 'while' loops for repeating while a condition is true."
            },
            "javascript": {
                "function": "A function is a block of code designed to perform a particular task. "
                          "JavaScript functions are first-class objects.",
                "class": "Classes are templates for creating objects in JavaScript. They encapsulate "
                        "data and behavior.",
                "promise": "A Promise represents a value that may be available now, in the future, or never."
            }
        }
        
        lang_explanations = explanations.get(language.value, {})
        return lang_explanations.get(concept.lower(), 
            f"A {concept} in {language.value} programming. Review the code for specific implementation details.")
    
    def _generate_examples(self, concept: str, language: CodeLanguage) -> List[Dict]:
        """Generate examples for the concept"""
        
        examples = {
            "python": {
                "function": [
                    {
                        "title": "Simple function",
                        "code": "def greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('Alice'))"
                    },
                    {
                        "title": "Function with parameters",
                        "code": "def add(a, b):\n    return a + b\n\nresult = add(5, 3)\nprint(result)  # Output: 8"
                    }
                ],
                "class": [
                    {
                        "title": "Simple class",
                        "code": "class Dog:\n    def __init__(self, name):\n        self.name = name\n    \n    def bark(self):\n        return f'{self.name} says woof!'"
                    }
                ]
            }
        }
        
        lang_examples = examples.get(language.value, {})
        return lang_examples.get(concept.lower(), [])
    
    def _identify_common_mistakes(self, concept: str, language: CodeLanguage) -> List[str]:
        """Identify common mistakes for the concept"""
        
        mistakes = {
            "python": {
                "function": [
                    "Forgetting to return a value from a function",
                    "Modifying mutable default arguments",
                    "Not handling exceptions properly"
                ],
                "class": [
                    "Forgetting 'self' parameter in method definitions",
                    "Not calling parent class __init__ in inheritance",
                    "Overusing class variables when instance variables are needed"
                ]
            }
        }
        
        lang_mistakes = mistakes.get(language.value, {})
        return lang_mistakes.get(concept.lower(), [])
    
    def _assess_difficulty(self, concept: str, language: CodeLanguage) -> str:
        """Assess difficulty level of the concept"""
        
        difficulty_map = {
            "python": {
                "function": "beginner",
                "class": "intermediate",
                "decorator": "advanced",
                "generator": "intermediate",
                "context manager": "intermediate"
            }
        }
        
        lang_map = difficulty_map.get(language.value, {})
        return lang_map.get(concept.lower(), "intermediate")
    
    def _identify_prerequisites(self, concept: str, language: CodeLanguage) -> List[str]:
        """Identify prerequisites for understanding the concept"""
        
        prerequisites = {
            "python": {
                "class": ["Understanding of functions", "Basic OOP concepts", "Python syntax"],
                "decorator": ["Functions", "First-class functions", "Closures"]
            }
        }
        
        lang_prereqs = prerequisites.get(language.value, {})
        return lang_prereqs.get(concept.lower(), [f"Basic {language.value} knowledge"])
    
    async def generate_tutorial(self, topic: str, difficulty: str,
                               language: CodeLanguage) -> Dict[str, Any]:
        """Generate step-by-step tutorial"""
        
        try:
            tutorial_steps = self._create_tutorial_steps(topic, difficulty, language)
            code_examples = self._generate_tutorial_code(topic, language)
            exercises = self._create_exercises(topic, difficulty, language)
            
            return {
                "topic": topic,
                "difficulty": difficulty,
                "language": language.value,
                "estimated_completion_time": self._estimate_tutorial_time(difficulty),
                "steps": tutorial_steps,
                "code_examples": code_examples,
                "exercises": exercises,
                "learning_objectives": self._define_learning_objectives(topic),
                "prerequisites": self._get_tutorial_prerequisites(topic, language)
            }
            
        except Exception as e:
            logger.error(f"Tutorial generation error: {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "language": language.value
            }
    
    def _create_tutorial_steps(self, topic: str, difficulty: str, 
                              language: CodeLanguage) -> List[Dict]:
        """Create tutorial steps"""
        
        steps = []
        
        # Sample steps structure
        if topic.lower() == "functions" and language == CodeLanguage.PYTHON:
            steps = [
                {
                    "step": 1,
                    "title": "Understanding Functions",
                    "content": "Learn what functions are and why we use them.",
                    "duration": "5 minutes"
                },
                {
                    "step": 2,
                    "title": "Defining Functions",
                    "content": "Learn the syntax for defining functions in Python.",
                    "duration": "10 minutes"
                },
                {
                    "step": 3,
                    "title": "Function Parameters",
                    "content": "Understand different types of parameters.",
                    "duration": "15 minutes"
                }
            ]
        
        return steps
    
    def _generate_tutorial_code(self, topic: str, language: CodeLanguage) -> List[Dict]:
        """Generate code examples for tutorial"""
        
        examples = []
        
        if topic.lower() == "functions" and language == CodeLanguage.PYTHON:
            examples = [
                {
                    "title": "Basic function definition",
                    "code": "def say_hello():\n    print('Hello, World!')\n\nsay_hello()",
                    "explanation": "Defines and calls a simple function."
                },
                {
                    "title": "Function with parameters",
                    "code": "def greet(name):\n    print(f'Hello, {name}!')\n\ngreet('Alice')",
                    "explanation": "Function that accepts a parameter."
                }
            ]
        
        return examples
    
    def _create_exercises(self, topic: str, difficulty: str, 
                         language: CodeLanguage) -> List[Dict]:
        """Create exercises for the tutorial"""
        
        exercises = []
        
        if topic.lower() == "functions" and language == CodeLanguage.PYTHON:
            exercises = [
                {
                    "title": "Write a calculator function",
                    "description": "Create a function that takes two numbers and an operation (+, -, *, /) and returns the result.",
                    "difficulty": "beginner",
                    "hint": "Use if/elif statements to handle different operations."
                }
            ]
        
        return exercises
    
    def _estimate_tutorial_time(self, difficulty: str) -> str:
        """Estimate tutorial completion time"""
        
        times = {
            "beginner": "30-45 minutes",
            "intermediate": "1-2 hours",
            "advanced": "2-4 hours"
        }
        
        return times.get(difficulty.lower(), "1 hour")
    
    def _define_learning_objectives(self, topic: str) -> List[str]:
        """Define learning objectives for the tutorial"""
        
        objectives = {
            "functions": [
                "Understand what functions are and why they're useful",
                "Learn how to define and call functions",
                "Understand function parameters and return values",
                "Practice writing your own functions"
            ]
        }
        
        return objectives.get(topic.lower(), ["Learn the basics of the topic"])
    
    def _get_tutorial_prerequisites(self, topic: str, language: CodeLanguage) -> List[str]:
        """Get prerequisites for the tutorial"""
        
        prereqs = {
            "python": {
                "functions": ["Basic Python syntax", "Variables and data types"],
                "classes": ["Functions", "Basic programming concepts"]
            }
        }
        
        lang_prereqs = prereqs.get(language.value, {})
        return lang_prereqs.get(topic.lower(), [f"Basic {language.value} knowledge"])

# ============================================================================
# MAIN ENGINE CLASS - PART 1 COMPLETE
# ============================================================================

class AICodeEngine:
    """Main AI Code Engine - Orchestrates all components"""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.model_router = ModelRouter(self.config)
        self.security_scanner = SecurityScanner(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.teaching_generator = TeachingResourceGeneratorImpl(self.config)
        self.cache = {}
        self.request_history = []
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the engine"""
        if not self.is_initialized:
            # Any async initialization would go here
            self.is_initialized = True
            logger.info("AI Code Engine initialized successfully")
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Main code generation endpoint"""
        
        await self.initialize()
        
        # Log request
        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "user_id": request.user_id,
            "session_id": request.session_id
        })
        
        # Limit history size
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
        
        # Generate code using model router
        response = await self.model_router.route_request(request)
        
        # Apply security scanning if requested
        if request.security_scan and response.success and response.code:
            try:
                security_scan = await self.security_scanner.scan_code(
                    response.code, request.language
                )
                response.security_scan_results = security_scan
                
                # Add security warnings if any
                if security_scan.get("vulnerabilities_found", 0) > 0:
                    response.warnings.append(
                        f"Found {security_scan['vulnerabilities_found']} security vulnerabilities"
                    )
                    
            except Exception as e:
                logger.error(f"Security scan failed: {str(e)}")
                response.warnings.append(f"Security scan failed: {str(e)}")
        
        # Apply performance optimization if requested
        if request.performance_optimize and response.success and response.code:
            try:
                optimization = await self.performance_optimizer.optimize_code(
                    response.code, request.language
                )
                response.performance_metrics = optimization
                
                # Update code if optimization provided better version
                if (optimization.get("optimization_score", 0) > 70 and 
                    optimization.get("optimized_code") and
                    len(optimization["optimized_code"]) > len(response.code) * 0.5):
                    
                    # Keep original but note optimization
                    response.suggestions.append(
                        f"Performance optimization available (score: {optimization['optimization_score']}/100)"
                    )
                    
            except Exception as e:
                logger.error(f"Performance optimization failed: {str(e)}")
        
        return response
    
    async def analyze_code(self, code: str, language: CodeLanguage,
                          analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze existing code"""
        
        await self.initialize()
        
        analysis_result = {}
        
        try:
            # Get model analysis
            model_analysis = await self.model_router.analyze_code(code, language)
            analysis_result["model_analysis"] = model_analysis
            
            # Get security analysis
            security_analysis = await self.security_scanner.scan_code(code, language)
            analysis_result["security_analysis"] = security_analysis
            
            # Get performance analysis
            performance_analysis = await self.performance_optimizer.benchmark_code(code, language)
            analysis_result["performance_analysis"] = performance_analysis
            
            # Combine results
            analysis_result["summary"] = self._generate_analysis_summary(
                model_analysis, security_analysis, performance_analysis
            )
            
            analysis_result["success"] = True
            analysis_result["analysis_type"] = analysis_type
            analysis_result["language"] = language.value
            analysis_result["timestamp"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            analysis_result = {
                "success": False,
                "error": str(e),
                "language": language.value
            }
        
        return analysis_result
    
    def _generate_analysis_summary(self, model_analysis: Dict, 
                                  security_analysis: Dict,
                                  performance_analysis: Dict) -> Dict[str, Any]:
        """Generate summary from analysis results"""
        
        summary = {
            "overall_quality": model_analysis.get("quality_score", 0),
            "security_score": security_analysis.get("security_score", 0),
            "performance_score": performance_analysis.get("code_metrics", {}).get("complexity_score", 0),
            "key_issues": [],
            "recommendations": []
        }
        
        # Combine issues
        if "issues" in model_analysis:
            summary["key_issues"].extend(model_analysis["issues"][:3])
        
        if "vulnerabilities" in security_analysis:
            vulns = security_analysis["vulnerabilities"][:2]
            for vuln in vulns:
                summary["key_issues"].append(f"Security: {vuln.get('type', 'Unknown')}")
        
        if "bottlenecks" in performance_analysis:
            summary["key_issues"].extend(performance_analysis["bottlenecks"][:2])
        
        # Combine recommendations
        if "recommendations" in model_analysis:
            summary["recommendations"].extend(model_analysis["recommendations"][:3])
        
        if "recommendations" in security_analysis:
            summary["recommendations"].extend(security_analysis["recommendations"][:2])
        
        # Calculate overall score
        scores = [
            summary["overall_quality"],
            summary["security_score"],
            summary["performance_score"]
        ]
        valid_scores = [s for s in scores if s > 0]
        
        if valid_scores:
            summary["overall_score"] = sum(valid_scores) // len(valid_scores)
        else:
            summary["overall_score"] = 0
        
        return summary
    
    async def explain_concept(self, concept: str, language: CodeLanguage,
                             detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain a programming concept"""
        
        await self.initialize()
        
        try:
            # Generate explanation
            explanation = await self.teaching_generator.generate_explanation(
                concept, "", language
            )
            
            # Get related code examples from model
            code_examples = await self._get_code_examples_for_concept(concept, language)
            
            explanation["code_examples"] = code_examples
            explanation["detail_level"] = detail_level
            explanation["success"] = True
            
            return explanation
            
        except Exception as e:
            logger.error(f"Concept explanation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "concept": concept,
                "language": language.value
            }
    
    async def _get_code_examples_for_concept(self, concept: str, 
                                            language: CodeLanguage) -> List[Dict]:
        """Get code examples for a concept using AI models"""
        
        try:
            # Use model router to generate examples
            example_prompt = f"Generate 2-3 {language.value} code examples demonstrating {concept}"
            
            request = CodeGenerationRequest(
                prompt=example_prompt,
                language=language,
                include_explanations=True,
                max_tokens=1000
            )
            
            response = await self.model_router.route_request(request)
            
            if response.success and response.code:
                # Parse examples from response
                examples = self._parse_examples_from_code(response.code, concept)
                return examples
            
        except Exception as e:
            logger.error(f"Failed to generate code examples: {str(e)}")
        
        return []
    
    def _parse_examples_from_code(self, code: str, concept: str) -> List[Dict]:
        """Parse examples from generated code"""
        
        examples = []
        
        # Simple parsing - split by double newlines
        code_blocks = code.split('\n\n')
        
        for i, block in enumerate(code_blocks[:3]):  # Limit to 3 examples
            if block.strip():
                examples.append({
                    "title": f"Example {i+1} - {concept}",
                    "code": block.strip(),
                    "explanation": f"Demonstrates {concept} implementation"
                })
        
        return examples
    
    async def create_tutorial(self, topic: str, difficulty: str,
                            language: CodeLanguage) -> Dict[str, Any]:
        """Create a programming tutorial"""
        
        await self.initialize()
        
        try:
            tutorial = await self.teaching_generator.generate_tutorial(
                topic, difficulty, language
            )
            
            tutorial["success"] = True
            tutorial["created_at"] = datetime.now().isoformat()
            
            return tutorial
            
        except Exception as e:
            logger.error(f"Tutorial creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic,
                "language": language.value
            }
    
    async def refactor_code(self, code: str, language: CodeLanguage,
                           optimization_target: str = "readability") -> Dict[str, Any]:
        """Refactor code with specific optimization target"""
        
        await self.initialize()
        
        try:
            refactoring_result = await self.model_router.refactor_code(
                code, language, optimization_target
            )
            
            # Add performance analysis
            if "refactored_code" in refactoring_result:
                performance_before = await self.performance_optimizer.benchmark_code(code, language)
                performance_after = await self.performance_optimizer.benchmark_code(
                    refactoring_result["refactored_code"], language
                )
                
                refactoring_result["performance_comparison"] = {
                    "before": performance_before,
                    "after": performance_after
                }
            
            refactoring_result["success"] = True
            refactoring_result["language"] = language.value
            refactoring_result["optimization_target"] = optimization_target
            
            return refactoring_result
            
        except Exception as e:
            logger.error(f"Code refactoring failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_code": code,
                "language": language.value,
                "optimization_target": optimization_target
            }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and metrics"""
        
        status = {
            "initialized": self.is_initialized,
            "models_available": list(self.model_router.models.keys()),
            "total_requests": len(self.request_history),
            "config": {
                "default_provider": self.config.config["engine"]["default_provider"],
                "caching_enabled": self.config.config["engine"]["enable_caching"],
                "security_enabled": self.config.config["security"]["enable_scanning"],
                "teaching_enabled": self.config.config["teaching"]["enable_explanations"]
            },
            "performance_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add model performance metrics
        for model_key, metrics in self.model_router.performance_metrics.items():
            status["performance_metrics"][model_key] = {
                "call_count": metrics.call_count,
                "success_rate": metrics.success_rate,
                "avg_confidence": metrics.avg_confidence,
                "last_used": metrics.last_used.isoformat()
            }
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        
        # Close sessions in models
        for model in self.model_router.models.values():
            if hasattr(model, 'session') and model.session:
                await model.session.close()
        
        self.is_initialized = False
        logger.info("AI Code Engine cleaned up")

# ============================================================================
# UTILITY FUNCTIONS FOR PART 2 & 3
# ============================================================================

# These functions will be used in Parts 2 and 3
# They're defined here to ensure availability across all parts

def validate_api_keys(config: EngineConfig) -> Dict[str, bool]:
    """Validate all API keys in configuration"""
    
    validation_results = {}
    
    for provider, provider_config in config.config["models"].items():
        api_key = provider_config.get("api_key", "")
        enabled = provider_config.get("enabled", False)
        
        if enabled:
            is_valid = len(api_key) > 10  # Simple validation
            validation_results[provider] = is_valid
    
    return validation_results

async def test_model_connectivity(engine: AICodeEngine) -> Dict[str, Any]:
    """Test connectivity to all configured models"""
    
    test_results = {}
    
    for model_key, model in engine.model_router.models.items():
        try:
            # Simple test request
            test_request = CodeGenerationRequest(
                prompt="Say 'Hello World'",
                language=CodeLanguage.PYTHON,
                max_tokens=10
            )
            
            start_time = datetime.now()
            response = await model.generate_code(test_request)
            response_time = (datetime.now() - start_time).total_seconds()
            
            test_results[model_key] = {
                "connected": response.success,
                "response_time": response_time,
                "error": response.error if not response.success else None
            }
            
        except Exception as e:
            test_results[model_key] = {
                "connected": False,
                "error": str(e)
            }
    
    return test_results

def generate_request_id() -> str:
    """Generate unique request ID"""
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    return f"req_{timestamp}_{random_part}"

async def benchmark_engine(engine: AICodeEngine, 
                          test_cases: List[CodeGenerationRequest]) -> Dict[str, Any]:
    """Benchmark engine performance with test cases"""
    
    benchmark_results = {
        "total_cases": len(test_cases),
        "results": [],
        "summary": {}
    }
    
    total_time = 0
    successful_calls = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            start_time = datetime.now()
            response = await engine.generate_code(test_case)
            response_time = (datetime.now() - start_time).total_seconds()
            
            total_time += response_time
            
            if response.success:
                successful_calls += 1
            
            benchmark_results["results"].append({
                "case_id": i,
                "success": response.success,
                "response_time": response_time,
                "confidence": response.confidence_score,
                "tokens_used": sum(response.token_usage.values()) if response.token_usage else 0
            })
            
        except Exception as e:
            benchmark_results["results"].append({
                "case_id": i,
                "success": False,
                "error": str(e)
            })
    
    # Calculate summary
    if benchmark_results["results"]:
        avg_response_time = total_time / len(benchmark_results["results"])
        success_rate = successful_calls / len(benchmark_results["results"])
        
        confidence_scores = [r.get("confidence", 0) for r in benchmark_results["results"] if r.get("confidence")]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        benchmark_results["summary"] = {
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "total_execution_time": total_time
        }
    
    return benchmark_results

# ============================================================================
# PART 1 COMPLETE
# Next parts will include:
# 1. Web API endpoints (FastAPI/Flask)
# 2. WebSocket for real-time communication  
# 3. Database integration for history/storage
# 4. Advanced caching strategies
# 5. Rate limiting and load balancing
# 6. Monitoring and analytics
# 7. Plugin system for extensibility
# 8. Frontend integration examples
# ============================================================================
```