```python
# File: advanced_ai_code_engine_part2.py
# PART 2/3: Web API, Database, Caching, Rate Limiting, and Monitoring

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import contextmanager
import jwt
from jwt.exceptions import InvalidTokenError
import aiohttp
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uuid
from functools import wraps
import cachetools
import zlib
import pickle

# secondary.py - FIX THESE IMPORTS:

# WRONG: (trying to import from non-existent module)
# from advanced_ai_code_engine_part2 import (...)

# CORRECT: (import from your actual file)
from main import (
    AICodeEngine, EngineConfig, CodeGenerationRequest, CodeGenerationResponse,
    CodeLanguage, ModelProvider, TaskComplexity, validate_api_keys,
    test_model_connectivity, generate_request_id, benchmark_engine
)

# Then continue with your secondary.py code...
# ============================================================================
# DATABASE MODELS
# ============================================================================

Base = declarative_base()

class User(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    rate_limit = Column(Integer, default=1000)  # requests per day
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Preferences
    default_language = Column(String(50), default="python")
    default_model = Column(String(50), default="ensemble")
    enable_teaching = Column(Boolean, default=True)
    enable_security_scan = Column(Boolean, default=True)
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    last_request_at = Column(DateTime, nullable=True)

class RequestHistory(Base):
    """Request history database model"""
    __tablename__ = "request_history"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=True)
    session_id = Column(String(100), index=True, nullable=True)
    
    # Request data
    prompt = Column(Text, nullable=False)
    language = Column(String(50), nullable=False)
    context = Column(Text, nullable=True)
    model_provider = Column(String(50), nullable=True)
    complexity = Column(String(50), default="medium")
    
    # Response data
    generated_code = Column(Text, nullable=True)
    success = Column(Boolean, default=False)
    execution_time = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    token_usage = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Analytics
    code_length = Column(Integer, default=0)
    has_warnings = Column(Boolean, default=False)
    has_errors = Column(Boolean, default=False)

class CodeAnalysis(Base):
    """Code analysis database model"""
    __tablename__ = "code_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=True)
    
    # Analysis data
    code_hash = Column(String(100), index=True, nullable=False)
    language = Column(String(50), nullable=False)
    
    # Quality metrics
    quality_score = Column(Integer, default=0)
    security_score = Column(Integer, default=0)
    performance_score = Column(Integer, default=0)
    readability_score = Column(Integer, default=0)
    
    # Issues
    vulnerabilities_found = Column(Integer, default=0)
    optimization_suggestions = Column(Integer, default=0)
    
    # Details
    analysis_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelUsage(Base):
    """Model usage tracking database model"""
    __tablename__ = "model_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), index=True, nullable=False)
    provider = Column(String(50), index=True, nullable=False)
    
    # Usage metrics
    request_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    avg_tokens = Column(Float, default=0.0)
    
    # Timestamps
    first_used = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    hour_bucket = Column(DateTime, index=True, nullable=False)  # For time-series analysis

class CacheEntry(Base):
    """Cache entry database model"""
    __tablename__ = "cache_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), unique=True, index=True, nullable=False)
    data_hash = Column(String(100), index=True, nullable=False)
    
    # Cache data
    data = Column(Text, nullable=False)  # JSON or compressed data
    data_type = Column(String(50), default="json")  # json, pickle, compressed
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    
    # Size info
    original_size = Column(Integer, default=0)
    compressed_size = Column(Integer, default=0)
    compression_ratio = Column(Float, default=1.0)

class Plugin(Base):
    """Plugin database model"""
    __tablename__ = "plugins"
    
    id = Column(Integer, primary_key=True, index=True)
    plugin_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    author = Column(String(100), nullable=True)
    
    # Plugin info
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)  # analysis, security, optimization, teaching
    enabled = Column(Boolean, default=True)
    
    # Plugin configuration
    config = Column(JSON, nullable=True)
    dependencies = Column(JSON, nullable=True)
    
    # Files
    main_module = Column(String(255), nullable=False)
    requirements = Column(Text, nullable=True)
    
    # Usage
    installed_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    usage_count = Column(Integer, default=0)

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class UserCreate(BaseModel):
    """User creation request model"""
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., regex=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    password: str = Field(..., min_length=8)
    
class UserResponse(BaseModel):
    """User response model"""
    id: int
    username: str
    email: str
    api_key: str
    is_active: bool
    is_admin: bool
    rate_limit: int
    total_requests: int
    total_tokens: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class CodeGenerationRequestAPI(BaseModel):
    """API request model for code generation"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="python")
    context: Optional[str] = Field(default=None, max_length=5000)
    temperature: Optional[float] = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4000, ge=1, le=32000)
    model_provider: Optional[str] = Field(default=None)
    complexity: Optional[str] = Field(default="medium")
    include_explanations: Optional[bool] = Field(default=True)
    include_tests: Optional[bool] = Field(default=False)
    security_scan: Optional[bool] = Field(default=True)
    performance_optimize: Optional[bool] = Field(default=False)
    session_id: Optional[str] = Field(default=None)
    
class CodeGenerationResponseAPI(BaseModel):
    """API response model for code generation"""
    request_id: str
    code: str
    success: bool
    model_used: str
    provider: str
    execution_time: float
    confidence_score: float
    token_usage: Optional[Dict[str, int]] = None
    warnings: List[str] = []
    suggestions: List[str] = []
    security_scan_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    teaching_resources: List[Dict[str, Any]] = []
    error: Optional[str] = None
    created_at: datetime

class CodeAnalysisRequest(BaseModel):
    """API request model for code analysis"""
    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(default="python")
    analysis_type: Optional[str] = Field(default="comprehensive")

class CodeAnalysisResponse(BaseModel):
    """API response model for code analysis"""
    analysis_id: str
    success: bool
    language: str
    analysis_type: str
    summary: Dict[str, Any]
    model_analysis: Optional[Dict[str, Any]] = None
    security_analysis: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime

class TutorialRequest(BaseModel):
    """API request model for tutorial generation"""
    topic: str = Field(..., min_length=1, max_length=200)
    difficulty: str = Field(default="beginner")
    language: str = Field(default="python")
    
class TutorialResponse(BaseModel):
    """API response model for tutorial generation"""
    tutorial_id: str
    success: bool
    topic: str
    difficulty: str
    language: str
    tutorial_data: Dict[str, Any]
    created_at: datetime

class PluginRegisterRequest(BaseModel):
    """API request model for plugin registration"""
    plugin_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="1.0.0")
    author: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    main_module: str = Field(...)
    requirements: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None

class PluginResponse(BaseModel):
    """API response model for plugins"""
    plugin_id: str
    name: str
    version: str
    author: Optional[str]
    description: Optional[str]
    category: Optional[str]
    enabled: bool
    config: Optional[Dict[str, Any]]
    installed_at: datetime
    usage_count: int

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Database management class"""
    
    def __init__(self, database_url: str = "sqlite:///ai_code_engine.db"):
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        logger.info(f"Database initialized at {database_url}")
    
    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_db(self):
        """Get database session for dependency injection"""
        with self.get_session() as session:
            yield session

# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Advanced cache manager with multiple backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enable_caching", True)
        self.cache_ttl = config.get("cache_ttl", 3600)
        
        # Initialize backends
        self.backends = {}
        
        # Memory cache (always available)
        self.memory_cache = cachetools.TTLCache(
            maxsize=config.get("memory_cache_size", 1000),
            ttl=self.cache_ttl
        )
        self.backends["memory"] = self.memory_cache
        
        # Redis cache (if configured)
        redis_config = config.get("redis", {})
        if redis_config.get("enabled", False):
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    password=redis_config.get("password", None),
                    db=redis_config.get("db", 0),
                    decode_responses=False  # Store bytes for compression
                )
                # Test connection
                self.redis_client.ping()
                self.backends["redis"] = self.redis_client
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis cache failed to initialize: {str(e)}")
        
        # Database cache (if configured)
        if config.get("database_cache", True):
            self.backends["database"] = "database"
        
        # Compression settings
        self.enable_compression = config.get("enable_compression", True)
        self.compression_threshold = config.get("compression_threshold", 1024)  # bytes
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        
        return f"cache_{hashlib.md5(data_str.encode()).hexdigest()}"
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib"""
        if len(data) < self.compression_threshold:
            return data
        
        try:
            compressed = zlib.compress(data, level=6)
            if len(compressed) < len(data):
                return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
        
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        try:
            return zlib.decompress(data)
        except:
            return data  # Not compressed or decompression failed
    
    def _serialize_data(self, data: Any) -> tuple:
        """Serialize data with automatic format detection"""
        # Try JSON first
        try:
            serialized = json.dumps(data).encode()
            data_type = "json"
            return serialized, data_type
        except:
            pass
        
        # Try pickle for complex objects
        try:
            serialized = pickle.dumps(data)
            data_type = "pickle"
            return serialized, data_type
        except Exception as e:
            logger.error(f"Serialization failed: {str(e)}")
            raise
    
    def _deserialize_data(self, data: bytes, data_type: str) -> Any:
        """Deserialize data based on format"""
        if data_type == "json":
            return json.loads(data.decode())
        elif data_type == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if not self.enabled:
            return None
        
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {key}")
            self.memory_cache[key]["last_accessed"] = datetime.utcnow()
            self.memory_cache[key]["hit_count"] += 1
            return self.memory_cache[key]["data"]
        
        # Try Redis
        if "redis" in self.backends:
            try:
                cached = self.redis_client.get(f"cache:{key}")
                if cached:
                    logger.debug(f"Cache hit (redis): {key}")
                    
                    # Parse stored data
                    cached_data = json.loads(cached.decode())
                    
                    # Check expiration
                    expires_at = cached_data.get("expires_at")
                    if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
                        self.redis_client.delete(f"cache:{key}")
                        return None
                    
                    data_bytes = base64.b64decode(cached_data["data"])
                    if cached_data.get("compressed", False):
                        data_bytes = self._decompress_data(data_bytes)
                    
                    data = self._deserialize_data(data_bytes, cached_data["data_type"])
                    
                    # Also store in memory cache for faster access
                    self.memory_cache[key] = {
                        "data": data,
                        "created_at": datetime.utcnow().isoformat(),
                        "expires_at": expires_at,
                        "hit_count": 1,
                        "last_accessed": datetime.utcnow()
                    }
                    
                    return data
            except Exception as e:
                logger.warning(f"Redis cache get failed: {str(e)}")
        
        # Try database cache
        if "database" in self.backends:
            try:
                with self.db_manager.get_session() as session:
                    cache_entry = session.query(CacheEntry).filter(
                        CacheEntry.cache_key == key,
                        (CacheEntry.expires_at == None) | (CacheEntry.expires_at > datetime.utcnow())
                    ).first()
                    
                    if cache_entry:
                        logger.debug(f"Cache hit (database): {key}")
                        
                        # Update access metrics
                        cache_entry.hit_count += 1
                        cache_entry.last_accessed = datetime.utcnow()
                        session.commit()
                        
                        # Load data
                        if cache_entry.data_type == "compressed":
                            data_bytes = self._decompress_data(base64.b64decode(cache_entry.data))
                        else:
                            data_bytes = base64.b64decode(cache_entry.data)
                        
                        data = self._deserialize_data(data_bytes, cache_entry.data_type.replace("_compressed", ""))
                        
                        # Store in memory cache
                        self.memory_cache[key] = {
                            "data": data,
                            "created_at": cache_entry.created_at.isoformat(),
                            "expires_at": cache_entry.expires_at.isoformat() if cache_entry.expires_at else None,
                            "hit_count": cache_entry.hit_count,
                            "last_accessed": datetime.utcnow()
                        }
                        
                        return data
            except Exception as e:
                logger.warning(f"Database cache get failed: {str(e)}")
        
        return None
    
    async def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set data in cache"""
        if not self.enabled:
            return False
        
        if ttl is None:
            ttl = self.cache_ttl
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
        
        # Serialize data
        try:
            serialized, data_type = self._serialize_data(data)
            original_size = len(serialized)
            
            # Compress if enabled and beneficial
            compressed = False
            if self.enable_compression and original_size >= self.compression_threshold:
                compressed_data = self._compress_data(serialized)
                if len(compressed_data) < original_size:
                    serialized = compressed_data
                    compressed = True
                    data_type = f"{data_type}_compressed"
        except Exception as e:
            logger.error(f"Cache serialization failed: {str(e)}")
            return False
        
        cache_entry = {
            "data": data,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "hit_count": 0,
            "last_accessed": datetime.utcnow()
        }
        
        success = True
        
        # Store in memory cache
        self.memory_cache[key] = cache_entry.copy()
        self.memory_cache[key]["data"] = data  # Keep original data
        
        # Store in Redis
        if "redis" in self.backends:
            try:
                cache_data = {
                    "data": base64.b64encode(serialized).decode(),
                    "data_type": data_type,
                    "compressed": compressed,
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "original_size": original_size,
                    "compressed_size": len(serialized)
                }
                
                redis_key = f"cache:{key}"
                if expires_at:
                    ttl_seconds = int((expires_at - datetime.utcnow()).total_seconds())
                    self.redis_client.setex(redis_key, ttl_seconds, json.dumps(cache_data))
                else:
                    self.redis_client.set(redis_key, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"Redis cache set failed: {str(e)}")
                success = False
        
        # Store in database
        if "database" in self.backends:
            try:
                with self.db_manager.get_session() as session:
                    db_entry = CacheEntry(
                        cache_key=key,
                        data_hash=hashlib.md5(serialized).hexdigest(),
                        data=base64.b64encode(serialized).decode(),
                        data_type=data_type,
                        created_at=datetime.utcnow(),
                        expires_at=expires_at,
                        original_size=original_size,
                        compressed_size=len(serialized),
                        compression_ratio=len(serialized) / max(original_size, 1)
                    )
                    session.add(db_entry)
                    session.commit()
            except Exception as e:
                logger.warning(f"Database cache set failed: {str(e)}")
                success = False
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete data from cache"""
        success = True
        
        # Delete from memory
        try:
            if key in self.memory_cache:
                del self.memory_cache[key]
        except:
            pass
        
        # Delete from Redis
        if "redis" in self.backends:
            try:
                self.redis_client.delete(f"cache:{key}")
            except:
                success = False
        
        # Delete from database
        if "database" in self.backends:
            try:
                with self.db_manager.get_session() as session:
                    session.query(CacheEntry).filter(CacheEntry.cache_key == key).delete()
                    session.commit()
            except:
                success = False
        
        return success
    
    async def clear(self, backend: Optional[str] = None) -> bool:
        """Clear cache"""
        success = True
        
        if backend is None or backend == "memory":
            try:
                self.memory_cache.clear()
            except:
                success = False
        
        if (backend is None or backend == "redis") and "redis" in self.backends:
            try:
                self.redis_client.flushdb()
            except:
                success = False
        
        if (backend is None or backend == "database") and "database" in self.backends:
            try:
                with self.db_manager.get_session() as session:
                    session.query(CacheEntry).delete()
                    session.commit()
            except:
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "enabled": self.enabled,
            "backends": list(self.backends.keys()),
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_maxsize": self.memory_cache.maxsize,
            "memory_cache_ttl": self.memory_cache.ttl
        }
        
        if "redis" in self.backends:
            try:
                redis_info = self.redis_client.info()
                stats["redis"] = {
                    "connected": True,
                    "used_memory": redis_info.get("used_memory_human", "N/A"),
                    "keys": self.redis_client.dbsize()
                }
            except:
                stats["redis"] = {"connected": False}
        
        if "database" in self.backends:
            try:
                with self.db_manager.get_session() as session:
                    total_entries = session.query(CacheEntry).count()
                    expired_entries = session.query(CacheEntry).filter(
                        CacheEntry.expires_at < datetime.utcnow()
                    ).count()
                    
                    stats["database"] = {
                        "total_entries": total_entries,
                        "expired_entries": expired_entries
                    }
            except:
                stats["database"] = {"error": "Unable to get stats"}
        
        return stats

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enable_rate_limiting", True)
        
        # Rate limit buckets
        self.buckets = {}
        
        # Initialize storage
        self.redis_client = None
        if config.get("redis", {}).get("enabled", False):
            try:
                redis_config = config["redis"]
                self.redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    password=redis_config.get("password", None),
                    db=redis_config.get("rate_limit_db", 1),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis rate limiter failed to initialize: {str(e)}")
    
    def _get_bucket_key(self, user_id: Optional[int], ip_address: str, endpoint: str) -> str:
        """Generate bucket key for rate limiting"""
        if user_id:
            return f"ratelimit:user:{user_id}:{endpoint}"
        else:
            return f"ratelimit:ip:{ip_address}:{endpoint}"
    
    async def check_rate_limit(self, 
                              user_id: Optional[int], 
                              ip_address: str, 
                              endpoint: str,
                              user_rate_limit: Optional[int] = None) -> Dict[str, Any]:
        """Check rate limit for user/IP"""
        
        if not self.enabled:
            return {"allowed": True, "remaining": -1, "reset": 0}
        
        # Get rate limits from config
        endpoint_limits = self.config.get("endpoint_limits", {})
        default_limit = self.config.get("default_limit", 60)  # requests per minute
        
        # Get limit for endpoint
        limit = endpoint_limits.get(endpoint, default_limit)
        
        # User-specific limits override
        if user_rate_limit:
            limit = min(limit, user_rate_limit)
        
        window = 60  # seconds
        
        # Generate bucket key
        bucket_key = self._get_bucket_key(user_id, ip_address, endpoint)
        
        current_time = time.time()
        window_start = current_time - window
        
        try:
            if self.redis_client:
                # Use Redis for distributed rate limiting
                pipe = self.redis_client.pipeline()
                
                # Remove old requests
                pipe.zremrangebyscore(bucket_key, 0, window_start)
                
                # Count current requests
                pipe.zcard(bucket_key)
                
                # Add current request
                pipe.zadd(bucket_key, {str(current_time): current_time})
                
                # Set expiry
                pipe.expire(bucket_key, window)
                
                results = pipe.execute()
                request_count = results[1]
                
                # Calculate remaining
                remaining = max(0, limit - request_count)
                reset_time = int(current_time + window)
                
                return {
                    "allowed": request_count <= limit,
                    "remaining": remaining,
                    "limit": limit,
                    "reset": reset_time,
                    "window": window
                }
            else:
                # Use in-memory rate limiting
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []
                
                # Remove old requests
                self.buckets[bucket_key] = [
                    req_time for req_time in self.buckets[bucket_key]
                    if req_time > window_start
                ]
                
                # Count current requests
                request_count = len(self.buckets[bucket_key])
                
                # Check if allowed
                if request_count >= limit:
                    # Find oldest request to calculate reset time
                    oldest_request = min(self.buckets[bucket_key]) if self.buckets[bucket_key] else current_time
                    reset_time = int(oldest_request + window)
                    
                    return {
                        "allowed": False,
                        "remaining": 0,
                        "limit": limit,
                        "reset": reset_time,
                        "window": window
                    }
                
                # Add current request
                self.buckets[bucket_key].append(current_time)
                
                # Calculate remaining
                remaining = limit - request_count - 1
                reset_time = int(current_time + window)
                
                return {
                    "allowed": True,
                    "remaining": remaining,
                    "limit": limit,
                    "reset": reset_time,
                    "window": window
                }
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            # Fail open - allow request if rate limiting fails
            return {"allowed": True, "remaining": -1, "reset": 0, "error": str(e)}
    
    def get_rate_limit_headers(self, rate_limit_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate rate limit headers for HTTP response"""
        return {
            "X-RateLimit-Limit": str(rate_limit_info.get("limit", 0)),
            "X-RateLimit-Remaining": str(rate_limit_info.get("remaining", 0)),
            "X-RateLimit-Reset": str(rate_limit_info.get("reset", 0)),
            "X-RateLimit-Window": str(rate_limit_info.get("window", 60))
        }
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get rate limit statistics for user"""
        if not self.redis_client:
            return {"error": "Redis not available for stats"}
        
        try:
            # Get all rate limit keys for user
            pattern = f"ratelimit:user:{user_id}:*"
            keys = self.redis_client.keys(pattern)
            
            stats = {}
            for key in keys:
                endpoint = key.split(":")[-1]
                request_count = self.redis_client.zcard(key)
                
                # Get window start time
                window_start = time.time() - 60
                recent_requests = self.redis_client.zcount(key, window_start, time.time())
                
                stats[endpoint] = {
                    "total_requests": request_count,
                    "recent_requests": recent_requests,
                    "key": key
                }
            
            return stats
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# AUTHENTICATION MANAGER
# ============================================================================

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.secret_key = config.get("secret_key", os.getenv("AUTH_SECRET_KEY", "default-secret-key-change-me"))
        self.algorithm = "HS256"
        self.token_expiry = config.get("token_expiry", 3600)  # 1 hour
        self.security = HTTPBearer()
        
        # Password hashing
        import bcrypt
        self.bcrypt = bcrypt
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = self.bcrypt.gensalt()
        hashed = self.bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    
    def create_access_token(self, user_id: int, username: str, is_admin: bool = False) -> str:
        """Create JWT access token"""
        payload = {
            "sub": str(user_id),
            "username": username,
            "is_admin": is_admin,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except InvalidTokenError as e:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return None
            if not self.verify_password(password, user.hashed_password):
                return None
            return user
    
    async def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> User:
        """Create new user"""
        with self.db_manager.get_session() as session:
            # Check if user exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise HTTPException(status_code=400, detail="Username or email already exists")
            
            # Create API key
            api_key = f"sk_{hashlib.sha256(f'{username}{email}{time.time()}'.encode()).hexdigest()[:32]}"
            
            # Create user
            user = User(
                username=username,
                email=email,
                hashed_password=self.hash_password(password),
                api_key=api_key,
                is_admin=is_admin
            )
            
            session.add(user)
            session.commit()
            session.refresh(user)
            
            return user
    
    async def get_current_user(self, 
                              credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
        """Get current user from token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        user_id = int(payload.get("sub"))
        
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            if not user.is_active:
                raise HTTPException(status_code=401, detail="User inactive")
            
            return user
    
    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        with self.db_manager.get_session() as session:
            return session.query(User).filter(User.api_key == api_key).first()

# ============================================================================
# MONITORING AND METRICS
# ============================================================================

class MetricsCollector:
    """Metrics collection for monitoring"""
    
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'engine_requests_total', 
            'Total requests', 
            ['endpoint', 'method', 'status']
        )
        
        self.request_duration = Histogram(
            'engine_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method']
        )
        
        # Model metrics
        self.model_requests = Counter(
            'engine_model_requests_total',
            'Total model requests',
            ['model', 'provider', 'status']
        )
        
        self.model_response_time = Histogram(
            'engine_model_response_time_seconds',
            'Model response time in seconds',
            ['model', 'provider']
        )
        
        # Code generation metrics
        self.code_generation_quality = Histogram(
            'engine_code_generation_quality',
            'Code generation quality score',
            ['language', 'complexity']
        )
        
        self.code_security_score = Histogram(
            'engine_code_security_score',
            'Code security score',
            ['language']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'engine_active_connections',
            'Number of active connections'
        )
        
        self.cache_hits = Counter(
            'engine_cache_hits_total',
            'Total cache hits',
            ['backend']
        )
        
        self.cache_misses = Counter(
            'engine_cache_misses_total',
            'Total cache misses',
            ['backend']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'engine_errors_total',
            'Total errors',
            ['type', 'endpoint']
        )
        
        # Token usage
        self.tokens_used = Counter(
            'engine_tokens_used_total',
            'Total tokens used',
            ['model', 'type']  # type: input/output/total
        )
    
    def record_request(self, endpoint: str, method: str, status: str, duration: float):
        """Record request metrics"""
        self.requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
        self.request_duration.labels(endpoint=endpoint, method=method).observe(duration)
    
    def record_model_request(self, model: str, provider: str, status: str, duration: float):
        """Record model request metrics"""
        self.model_requests.labels(model=model, provider=provider, status=status).inc()
        self.model_response_time.labels(model=model, provider=provider).observe(duration)
    
    def record_code_quality(self, language: str, complexity: str, score: float):
        """Record code quality metrics"""
        self.code_generation_quality.labels(language=language, complexity=complexity).observe(score)
    
    def record_security_score(self, language: str, score: float):
        """Record security score metrics"""
        self.code_security_score.labels(language=language).observe(score)
    
    def record_cache_hit(self, backend: str):
        """Record cache hit"""
        self.cache_hits.labels(backend=backend).inc()
    
    def record_cache_miss(self, backend: str):
        """Record cache miss"""
        self.cache_misses.labels(backend=backend).inc()
    
    def record_error(self, error_type: str, endpoint: str):
        """Record error"""
        self.errors_total.labels(type=error_type, endpoint=endpoint).inc()
    
    def record_tokens(self, model: str, token_type: str, count: int):
        """Record token usage"""
        self.tokens_used.labels(model=model, type=token_type).inc(count)
    
    def set_active_connections(self, count: int):
        """Set active connections count"""
        self.active_connections.set(count)
    
    def get_metrics(self):
        """Get Prometheus metrics"""
        return generate_latest()

# ============================================================================
# PLUGIN SYSTEM
# ============================================================================

class PluginManager:
    """Plugin manager for extensibility"""
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.plugins = {}
        self.plugin_categories = {
            "analysis": [],
            "security": [],
            "optimization": [],
            "teaching": [],
            "custom": []
        }
        
        # Load plugins
        self.load_plugins()
    
    def load_plugins(self):
        """Load plugins from database and filesystem"""
        try:
            with self.db_manager.get_session() as session:
                db_plugins = session.query(Plugin).filter(Plugin.enabled == True).all()
                
                for db_plugin in db_plugins:
                    try:
                        self._load_plugin(db_plugin)
                        logger.info(f"Loaded plugin: {db_plugin.name} v{db_plugin.version}")
                    except Exception as e:
                        logger.error(f"Failed to load plugin {db_plugin.name}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to load plugins: {str(e)}")
    
    def _load_plugin(self, db_plugin: Plugin):
        """Load individual plugin"""
        plugin_id = db_plugin.plugin_id
        
        # Dynamically import plugin module
        import importlib.util
        import sys
        
        # Try to import from plugin directory
        plugin_dir = self.config.get("plugin_dir", "plugins")
        plugin_path = os.path.join(plugin_dir, f"{plugin_id}.py")
        
        if not os.path.exists(plugin_path):
            # Try to find main module
            plugin_path = db_plugin.main_module
            if not os.path.exists(plugin_path):
                raise FileNotFoundError(f"Plugin module not found: {plugin_path}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(f"plugins.{plugin_id}", plugin_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"plugins.{plugin_id}"] = module
        spec.loader.exec_module(module)
        
        # Get plugin class (should be named PluginClass)
        plugin_class = getattr(module, "PluginClass", None)
        if not plugin_class:
            raise AttributeError("PluginClass not found in module")
        
        # Instantiate plugin
        plugin_instance = plugin_class(db_plugin.config or {})
        
        # Store plugin
        self.plugins[plugin_id] = {
            "instance": plugin_instance,
            "db_info": db_plugin,
            "module": module
        }
        
        # Categorize
        category = db_plugin.category or "custom"
        if category in self.plugin_categories:
            self.plugin_categories[category].append(plugin_id)
    
    async def execute_plugin(self, plugin_id: str, action: str, data: Any, context: Dict[str, Any]) -> Any:
        """Execute plugin action"""
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_id}")
        
        plugin = self.plugins[plugin_id]
        plugin_instance = plugin["instance"]
        
        # Check if action exists
        if not hasattr(plugin_instance, action):
            raise AttributeError(f"Action {action} not found in plugin {plugin_id}")
        
        # Update usage count
        with self.db_manager.get_session() as session:
            db_plugin = session.query(Plugin).filter(Plugin.plugin_id == plugin_id).first()
            if db_plugin:
                db_plugin.usage_count += 1
                session.commit()
        
        # Execute action
        method = getattr(plugin_instance, action)
        
        if asyncio.iscoroutinefunction(method):
            result = await method(data, context)
        else:
            result = method(data, context)
        
        return result
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin information"""
        if plugin_id not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_id]
        db_plugin = plugin["db_info"]
        
        return {
            "plugin_id": db_plugin.plugin_id,
            "name": db_plugin.name,
            "version": db_plugin.version,
            "author": db_plugin.author,
            "description": db_plugin.description,
            "category": db_plugin.category,
            "enabled": db_plugin.enabled,
            "config": db_plugin.config,
            "installed_at": db_plugin.installed_at,
            "usage_count": db_plugin.usage_count,
            "available_actions": self._get_plugin_actions(plugin["instance"])
        }
    
    def _get_plugin_actions(self, plugin_instance: Any) -> List[str]:
        """Get available actions from plugin instance"""
        actions = []
        
        # Get all methods that don't start with underscore
        for attr_name in dir(plugin_instance):
            if not attr_name.startswith("_"):
                attr = getattr(plugin_instance, attr_name)
                if callable(attr):
                    actions.append(attr_name)
        
        return actions
    
    async def register_plugin(self, plugin_data: Dict[str, Any]) -> Plugin:
        """Register new plugin"""
        with self.db_manager.get_session() as session:
            # Check if plugin exists
            existing = session.query(Plugin).filter(Plugin.plugin_id == plugin_data["plugin_id"]).first()
            if existing:
                raise HTTPException(status_code=400, detail="Plugin already exists")
            
            # Create plugin
            plugin = Plugin(
                plugin_id=plugin_data["plugin_id"],
                name=plugin_data["name"],
                version=plugin_data.get("version", "1.0.0"),
                author=plugin_data.get("author"),
                description=plugin_data.get("description"),
                category=plugin_data.get("category"),
                main_module=plugin_data["main_module"],
                requirements=plugin_data.get("requirements"),
                config=plugin_data.get("config"),
                dependencies=plugin_data.get("dependencies")
            )
            
            session.add(plugin)
            session.commit()
            session.refresh(plugin)
            
            # Try to load the plugin
            try:
                self._load_plugin(plugin)
            except Exception as e:
                logger.error(f"Failed to load plugin after registration: {str(e)}")
                # Don't fail registration, just log error
            
            return plugin
    
    def get_all_plugins(self) -> List[Dict[str, Any]]:
        """Get all plugins"""
        plugins_info = []
        
        for plugin_id in self.plugins:
            info = self.get_plugin_info(plugin_id)
            if info:
                plugins_info.append(info)
        
        return plugins_info

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class WebSocketManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_data[client_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "user_id": None,
            "session_id": None
        }
    
    def disconnect(self, client_id: str):
        """Disconnect WebSocket"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_data:
            del self.connection_data[client_id]
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
                self.connection_data[client_id]["last_activity"] = datetime.utcnow()
            except:
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude: List[str] = None):
        """Broadcast message to all connected clients"""
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if exclude and client_id in exclude:
                continue
            
            try:
                await websocket.send_json(message)
                self.connection_data[client_id]["last_activity"] = datetime.utcnow()
            except:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        now = datetime.utcnow()
        
        stats = {
            "total_connections": len(self.active_connections),
            "active_connections": list(self.active_connections.keys()),
            "connection_details": {}
        }
        
        for client_id, data in self.connection_data.items():
            stats["connection_details"][client_id] = {
                "connected_for": str(now - data["connected_at"]),
                "last_activity": data["last_activity"],
                "user_id": data["user_id"],
                "session_id": data["session_id"]
            }
        
        return stats

# ============================================================================
# MAIN API APPLICATION
# ============================================================================

class AICodeEngineAPI:
    """Main API application class"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize core engine
        self.engine_config = EngineConfig()
        self.engine = AICodeEngine(self.engine_config)
        
        # Initialize database
        database_url = self.config.get("database_url", "sqlite:///ai_code_engine.db")
        self.db_manager = DatabaseManager(database_url)
        
        # Initialize cache manager
        cache_config = self.config.get("cache", {})
        self.cache_manager = CacheManager(cache_config)
        self.cache_manager.db_manager = self.db_manager
        
        # Initialize rate limiter
        rate_limit_config = self.config.get("rate_limiting", {})
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Initialize authentication
        auth_config = self.config.get("authentication", {})
        self.auth_manager = AuthManager(auth_config, self.db_manager)
        
        # Initialize metrics
        self.metrics = MetricsCollector()
        
        # Initialize plugin manager
        plugin_config = self.config.get("plugins", {})
        self.plugin_manager = PluginManager(plugin_config, self.db_manager)
        
        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AI Code Generation Engine API",
            description="Advanced multi-model AI code generation engine with teaching capabilities",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configure CORS
        self._configure_cors()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.background_tasks = set()
        
        logger.info("AI Code Engine API initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {
            "database_url": os.getenv("DATABASE_URL", "sqlite:///ai_code_engine.db"),
            "cache": {
                "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
                "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
                "memory_cache_size": int(os.getenv("MEMORY_CACHE_SIZE", "1000")),
                "redis": {
                    "enabled": os.getenv("REDIS_ENABLED", "false").lower() == "true",
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6379")),
                    "password": os.getenv("REDIS_PASSWORD", None),
                    "db": int(os.getenv("REDIS_DB", "0"))
                },
                "database_cache": os.getenv("DATABASE_CACHE", "true").lower() == "true",
                "enable_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
                "compression_threshold": int(os.getenv("COMPRESSION_THRESHOLD", "1024"))
            },
            "rate_limiting": {
                "enable_rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
                "default_limit": int(os.getenv("RATE_LIMIT_DEFAULT", "60")),
                "endpoint_limits": {
                    "generate": int(os.getenv("RATE_LIMIT_GENERATE", "30")),
                    "analyze": int(os.getenv("RATE_LIMIT_ANALYZE", "60")),
                    "explain": int(os.getenv("RATE_LIMIT_EXPLAIN", "30"))
                },
                "redis": {
                    "enabled": os.getenv("REDIS_RATE_LIMIT_ENABLED", "false").lower() == "true",
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6379")),
                    "password": os.getenv("REDIS_PASSWORD", None),
                    "rate_limit_db": int(os.getenv("REDIS_RATE_LIMIT_DB", "1"))
                }
            },
            "authentication": {
                "secret_key": os.getenv("AUTH_SECRET_KEY", "default-secret-key-change-me"),
                "token_expiry": int(os.getenv("TOKEN_EXPIRY", "3600"))
            },
            "plugins": {
                "plugin_dir": os.getenv("PLUGIN_DIR", "plugins"),
                "auto_load": os.getenv("PLUGIN_AUTO_LOAD", "true").lower() == "true"
            },
            "cors": {
                "origins": os.getenv("CORS_ORIGINS", "*").split(","),
                "methods": os.getenv("CORS_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(","),
                "headers": os.getenv("CORS_HEADERS", "*").split(",")
            }
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {str(e)}")
        
        return config
    
    def _configure_cors(self):
        """Configure CORS middleware"""
        cors_config = self.config.get("cors", {})
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=cors_config.get("methods", ["*"]),
            allow_headers=cors_config.get("headers", ["*"])
        )
    
    def _setup_middleware(self):
        """Setup custom middleware"""
        
        @self.app.middleware("http")
        async def metrics_middleware(request, call_next):
            """Middleware for collecting metrics"""
            start_time = time.time()
            endpoint = request.url.path
            method = request.method
            
            try:
                response = await call_next(request)
                status = "success" if response.status_code < 400 else "error"
                duration = time.time() - start_time
                
                self.metrics.record_request(endpoint, method, status, duration)
                
                return response
            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_request(endpoint, method, "error", duration)
                self.metrics.record_error(type(e).__name__, endpoint)
                raise
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            """Middleware for rate limiting"""
            # Skip rate limiting for certain endpoints
            if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)
            
            # Get user from API key if provided
            user_id = None
            user_rate_limit = None
            
            api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
            if api_key:
                if api_key.startswith("Bearer "):
                    api_key = api_key[7:]
                elif api_key.startswith("ApiKey "):
                    api_key = api_key[7:]
                
                user = await self.auth_manager.get_user_by_api_key(api_key)
                if user:
                    user_id = user.id
                    user_rate_limit = user.rate_limit
            
            # Get IP address
            ip_address = request.client.host if request.client else "unknown"
            
            # Check rate limit
            endpoint = request.url.path.split("/")[-1] if "/" in request.url.path else request.url.path
            rate_limit_info = await self.rate_limiter.check_rate_limit(
                user_id, ip_address, endpoint, user_rate_limit
            )
            
            if not rate_limit_info.get("allowed", True):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers=self.rate_limiter.get_rate_limit_headers(rate_limit_info)
                )
            
            response = await call_next(request)
            
            # Add rate limit headers to response
            for key, value in self.rate_limiter.get_rate_limit_headers(rate_limit_info).items():
                response.headers[key] = value
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0"
            }
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            from starlette.responses import Response
            return Response(
                content=self.metrics.get_metrics(),
                media_type="text/plain"
            )
        
        # Authentication routes
        @self.app.post("/auth/register", response_model=UserResponse)
        async def register_user(user_data: UserCreate):
            """Register new user"""
            try:
                user = await self.auth_manager.create_user(
                    username=user_data.username,
                    email=user_data.email,
                    password=user_data.password
                )
                return user
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/login")
        async def login_user(username: str, password: str):
            """Login user and get token"""
            user = await self.auth_manager.authenticate_user(username, password)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            token = self.auth_manager.create_access_token(
                user_id=user.id,
                username=user.username,
                is_admin=user.is_admin
            )
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "user_id": user.id,
                "username": user.username,
                "is_admin": user.is_admin
            }
        
        # Code generation endpoint
        @self.app.post("/generate", response_model=CodeGenerationResponseAPI)
        async def generate_code(
            request_data: CodeGenerationRequestAPI,
            background_tasks: BackgroundTasks,
            user: Optional[User] = Depends(self.auth_manager.get_current_user)
        ):
            """Generate code from natural language prompt"""
            start_time = time.time()
            request_id = generate_request_id()
            
            try:
                # Convert API request to engine request
                engine_request = CodeGenerationRequest(
                    prompt=request_data.prompt,
                    language=CodeLanguage(request_data.language),
                    context=request_data.context,
                    temperature=request_data.temperature,
                    max_tokens=request_data.max_tokens,
                    model_provider=ModelProvider(request_data.model_provider) if request_data.model_provider else None,
                    complexity=TaskComplexity(request_data.complexity),
                    include_explanations=request_data.include_explanations,
                    include_tests=request_data.include_tests,
                    security_scan=request_data.security_scan,
                    performance_optimize=request_data.performance_optimize,
                    session_id=request_data.session_id,
                    user_id=user.id if user else None
                )
                
                # Check cache first
                cache_key = self.cache_manager._generate_cache_key({
                    "prompt": request_data.prompt[:200],
                    "language": request_data.language,
                    "context_hash": hashlib.md5((request_data.context or "").encode()).hexdigest()[:8],
                    "temperature": request_data.temperature,
                    "model": request_data.model_provider or "auto"
                })
                
                cached_response = await self.cache_manager.get(cache_key)
                if cached_response:
                    self.metrics.record_cache_hit("memory")
                    logger.info(f"Serving cached response for {request_id}")
                    
                    # Update cached response with request_id
                    cached_response.request_id = request_id
                    cached_response.created_at = datetime.utcnow()
                    
                    # Record metrics
                    self.metrics.record_model_request(
                        cached_response.model_used,
                        cached_response.provider,
                        "success" if cached_response.success else "error",
                        cached_response.execution_time
                    )
                    
                    return cached_response
                
                self.metrics.record_cache_miss("memory")
                
                # Generate code using engine
                response = await self.engine.generate_code(engine_request)
                
                # Add request ID
                response.request_id = request_id
                
                # Record model metrics
                self.metrics.record_model_request(
                    response.model_used,
                    response.provider,
                    "success" if response.success else "error",
                    response.execution_time
                )
                
                # Record quality metrics
                if response.success:
                    self.metrics.record_code_quality(
                        request_data.language,
                        request_data.complexity,
                        response.confidence_score
                    )
                
                # Record token usage
                if response.token_usage:
                    for token_type, count in response.token_usage.items():
                        self.metrics.record_tokens(response.model_used, token_type, count)
                
                # Cache successful responses
                if response.success:
                    background_tasks.add_task(
                        self.cache_manager.set,
                        cache_key,
                        response,
                        ttl=self.cache_manager.cache_ttl
                    )
                
                # Save to database
                background_tasks.add_task(
                    self._save_request_to_db,
                    request_id,
                    user.id if user else None,
                    engine_request,
                    response,
                    request_data.session_id
                )
                
                # Update user statistics
                if user:
                    background_tasks.add_task(
                        self._update_user_stats,
                        user.id,
                        response
                    )
                
                # Send WebSocket notification
                if user and response.success:
                    background_tasks.add_task(
                        self._notify_websocket_clients,
                        user.id,
                        request_data.session_id,
                        {
                            "type": "code_generated",
                            "request_id": request_id,
                            "success": response.success,
                            "execution_time": response.execution_time
                        }
                    )
                
                return response
                
            except Exception as e:
                self.metrics.record_error(type(e).__name__, "/generate")
                logger.error(f"Code generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Code analysis endpoint
        @self.app.post("/analyze", response_model=CodeAnalysisResponse)
        async def analyze_code(
            request_data: CodeAnalysisRequest,
            user: Optional[User] = Depends(self.auth_manager.get_current_user)
        ):
            """Analyze existing code"""
            try:
                analysis_id = f"analysis_{generate_request_id()}"
                
                # Perform analysis
                analysis_result = await self.engine.analyze_code(
                    request_data.code,
                    CodeLanguage(request_data.language),
                    request_data.analysis_type
                )
                
                # Record security score
                if "security_analysis" in analysis_result:
                    security_score = analysis_result["security_analysis"].get("security_score", 0)
                    self.metrics.record_security_score(request_data.language, security_score)
                
                # Create response
                response = CodeAnalysisResponse(
                    analysis_id=analysis_id,
                    success=analysis_result.get("success", False),
                    language=request_data.language,
                    analysis_type=request_data.analysis_type,
                    summary=analysis_result.get("summary", {}),
                    model_analysis=analysis_result.get("model_analysis"),
                    security_analysis=analysis_result.get("security_analysis"),
                    performance_analysis=analysis_result.get("performance_analysis"),
                    created_at=datetime.utcnow()
                )
                
                # Save to database
                background_tasks = BackgroundTasks()
                background_tasks.add_task(
                    self._save_analysis_to_db,
                    analysis_id,
                    user.id if user else None,
                    request_data,
                    analysis_result
                )
                
                return response
                
            except Exception as e:
                self.metrics.record_error(type(e).__name__, "/analyze")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Tutorial generation endpoint
        @self.app.post("/tutorial", response_model=TutorialResponse)
        async def generate_tutorial(
            request_data: TutorialRequest,
            user: Optional[User] = Depends(self.auth_manager.get_current_user)
        ):
            """Generate programming tutorial"""
            try:
                tutorial_id = f"tutorial_{generate_request_id()}"
                
                # Generate tutorial
                tutorial_result = await self.engine.create_tutorial(
                    request_data.topic,
                    request_data.difficulty,
                    CodeLanguage(request_data.language)
                )
                
                # Create response
                response = TutorialResponse(
                    tutorial_id=tutorial_id,
                    success=tutorial_result.get("success", False),
                    topic=request_data.topic,
                    difficulty=request_data.difficulty,
                    language=request_data.language,
                    tutorial_data=tutorial_result,
                    created_at=datetime.utcnow()
                )
                
                return response
                
            except Exception as e:
                self.metrics.record_error(type(e).__name__, "/tutorial")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Plugin endpoints
        @self.app.get("/plugins", response_model=List[PluginResponse])
        async def list_plugins(
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """List all available plugins"""
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            plugins = self.plugin_manager.get_all_plugins()
            return plugins
        
        @self.app.post("/plugins/register", response_model=PluginResponse)
        async def register_plugin(
            plugin_data: PluginRegisterRequest,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Register new plugin"""
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                plugin = await self.plugin_manager.register_plugin(plugin_data.dict())
                return plugin
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/plugins/{plugin_id}/execute")
        async def execute_plugin(
            plugin_id: str,
            action: str,
            data: Dict[str, Any],
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Execute plugin action"""
            try:
                context = {"user_id": user.id, "username": user.username}
                result = await self.plugin_manager.execute_plugin(plugin_id, action, data, context)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # System endpoints
        @self.app.get("/system/status")
        async def system_status(
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Get system status"""
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            status = {
                "engine": self.engine.get_engine_status(),
                "cache": self.cache_manager.get_stats(),
                "websocket": self.websocket_manager.get_connection_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return status
        
        @self.app.get("/system/analytics")
        async def system_analytics(
            timeframe: str = "24h",
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Get system analytics"""
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            analytics = await self._get_system_analytics(timeframe)
            return analytics
        
        @self.app.post("/system/cache/clear")
        async def clear_cache(
            backend: Optional[str] = None,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Clear cache"""
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            success = await self.cache_manager.clear(backend)
            return {"success": success, "backend": backend or "all"}
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time updates"""
            await self.websocket_manager.connect(websocket, client_id)
            
            try:
                while True:
                    # Keep connection alive
                    data = await websocket.receive_text()
                    
                    # Handle incoming messages
                    try:
                        message = json.loads(data)
                        await self._handle_websocket_message(client_id, message)
                    except json.JSONDecodeError:
                        pass
                        
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(client_id)
    
    async def _save_request_to_db(self, 
                                 request_id: str,
                                 user_id: Optional[int],
                                 request: CodeGenerationRequest,
                                 response: CodeGenerationResponse,
                                 session_id: Optional[str]):
        """Save request to database"""
        try:
            with self.db_manager.get_session() as session:
                # Save request history
                history = RequestHistory(
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    prompt=request.prompt[:5000],  # Limit size
                    language=request.language.value,
                    context=request.context[:2000] if request.context else None,
                    model_provider=request.model_provider.value if request.model_provider else None,
                    complexity=request.complexity.value,
                    generated_code=response.code[:10000] if response.code else None,
                    success=response.success,
                    execution_time=response.execution_time,
                    confidence_score=response.confidence_score,
                    token_usage=response.token_usage,
                    code_length=len(response.code) if response.code else 0,
                    has_warnings=len(response.warnings) > 0,
                    has_errors=response.error is not None,
                    created_at=datetime.utcnow()
                )
                
                session.add(history)
                
                # Update model usage statistics
                if response.success:
                    hour_bucket = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                    
                    model_usage = session.query(ModelUsage).filter(
                        ModelUsage.model_name == response.model_used,
                        ModelUsage.provider == response.provider.value,
                        ModelUsage.hour_bucket == hour_bucket
                    ).first()
                    
                    if not model_usage:
                        model_usage = ModelUsage(
                            model_name=response.model_used,
                            provider=response.provider.value,
                            hour_bucket=hour_bucket,
                            first_used=datetime.utcnow()
                        )
                        session.add(model_usage)
                    
                    # Update metrics
                    model_usage.request_count += 1
                    model_usage.success_count += 1
                    model_usage.last_used = datetime.utcnow()
                    
                    # Update averages
                    total_requests = model_usage.request_count
                    model_usage.avg_response_time = (
                        (model_usage.avg_response_time * (total_requests - 1) + response.execution_time) / total_requests
                    )
                    model_usage.avg_confidence = (
                        (model_usage.avg_confidence * (total_requests - 1) + response.confidence_score) / total_requests
                    )
                    
                    if response.token_usage:
                        total_tokens = sum(response.token_usage.values())
                        model_usage.avg_tokens = (
                            (model_usage.avg_tokens * (total_requests - 1) + total_tokens) / total_requests
                        )
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to save request to database: {str(e)}")
    
    async def _save_analysis_to_db(self,
                                  analysis_id: str,
                                  user_id: Optional[int],
                                  request: CodeAnalysisRequest,
                                  analysis_result: Dict[str, Any]):
        """Save analysis to database"""
        try:
            with self.db_manager.get_session() as session:
                code_hash = hashlib.md5(request.code.encode()).hexdigest()
                
                analysis = CodeAnalysis(
                    request_id=analysis_id,
                    user_id=user_id,
                    code_hash=code_hash,
                    language=request.language,
                    quality_score=analysis_result.get("summary", {}).get("overall_quality", 0),
                    security_score=analysis_result.get("summary", {}).get("security_score", 0),
                    performance_score=analysis_result.get("summary", {}).get("performance_score", 0),
                    readability_score=analysis_result.get("model_analysis", {}).get("readability_score", 0),
                    vulnerabilities_found=analysis_result.get("security_analysis", {}).get("vulnerabilities_found", 0),
                    optimization_suggestions=len(analysis_result.get("performance_analysis", {}).get("bottlenecks", [])),
                    analysis_data=analysis_result,
                    created_at=datetime.utcnow()
                )
                
                session.add(analysis)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {str(e)}")
    
    async def _update_user_stats(self, user_id: int, response: CodeGenerationResponse):
        """Update user statistics"""
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user:
                    user.total_requests += 1
                    user.last_request_at = datetime.utcnow()
                    
                    if response.token_usage:
                        total_tokens = sum(response.token_usage.values())
                        user.total_tokens += total_tokens
                    
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update user stats: {str(e)}")
    
    async def _notify_websocket_clients(self, user_id: int, session_id: Optional[str], message: Dict[str, Any]):
        """Notify WebSocket clients"""
        try:
            # Find connections for this user/session
            for client_id, data in self.websocket_manager.connection_data.items():
                if data["user_id"] == user_id or data["session_id"] == session_id:
                    await self.websocket_manager.send_message(client_id, message)
                    
        except Exception as e:
            logger.error(f"Failed to notify WebSocket clients: {str(e)}")
    
    async def _handle_websocket_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            
            if message_type == "authenticate":
                # Authenticate WebSocket connection
                token = message.get("token")
                if token:
                    try:
                        payload = self.auth_manager.verify_token(token)
                        user_id = int(payload.get("sub"))
                        
                        with self.db_manager.get_session() as session:
                            user = session.query(User).filter(User.id == user_id).first()
                            if user:
                                self.websocket_manager.connection_data[client_id]["user_id"] = user_id
                                
                                await self.websocket_manager.send_message(client_id, {
                                    "type": "authentication_success",
                                    "user_id": user_id,
                                    "username": user.username
                                })
                    except:
                        pass
            
            elif message_type == "subscribe":
                # Subscribe to updates
                session_id = message.get("session_id")
                if session_id:
                    self.websocket_manager.connection_data[client_id]["session_id"] = session_id
                    
                    await self.websocket_manager.send_message(client_id, {
                        "type": "subscription_success",
                        "session_id": session_id
                    })
                    
        except Exception as e:
            logger.error(f"Failed to handle WebSocket message: {str(e)}")
    
    async def _get_system_analytics(self, timeframe: str) -> Dict[str, Any]:
        """Get system analytics for given timeframe"""
        try:
            with self.db_manager.get_session() as session:
                # Calculate time range
                now = datetime.utcnow()
                if timeframe == "1h":
                    start_time = now - timedelta(hours=1)
                elif timeframe == "24h":
                    start_time = now - timedelta(days=1)
                elif timeframe == "7d":
                    start_time = now - timedelta(days=7)
                elif timeframe == "30d":
                    start_time = now - timedelta(days=30)
                else:
                    start_time = now - timedelta(days=1)
                
                # Get request statistics
                total_requests = session.query(RequestHistory).filter(
                    RequestHistory.created_at >= start_time
                ).count()
                
                successful_requests = session.query(RequestHistory).filter(
                    RequestHistory.created_at >= start_time,
                    RequestHistory.success == True
                ).count()
                
                failed_requests = total_requests - successful_requests
                
                # Get user statistics
                active_users = session.query(RequestHistory.user_id).filter(
                    RequestHistory.created_at >= start_time
                ).distinct().count()
                
                # Get model usage
                model_usage = session.query(
                    ModelUsage.model_name,
                    ModelUsage.provider,
                    ModelUsage.request_count,
                    ModelUsage.success_count,
                    ModelUsage.avg_response_time,
                    ModelUsage.avg_confidence
                ).filter(
                    ModelUsage.hour_bucket >= start_time
                ).all()
                
                # Get language distribution
                language_dist = session.query(
                    RequestHistory.language,
                    func.count(RequestHistory.id).label('count')
                ).filter(
                    RequestHistory.created_at >= start_time
                ).group_by(RequestHistory.language).all()
                
                # Get hourly request distribution
                hourly_dist = []
                for hour in range(24):
                    hour_start = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    hour_end = hour_start + timedelta(hours=1)
                    
                    if hour_start >= start_time:
                        hour_count = session.query(RequestHistory).filter(
                            RequestHistory.created_at >= hour_start,
                            RequestHistory.created_at < hour_end
                        ).count()
                        
                        hourly_dist.append({
                            "hour": hour,
                            "count": hour_count
                        })
                
                # Compile analytics
                analytics = {
                    "timeframe": timeframe,
                    "period": {
                        "start": start_time.isoformat(),
                        "end": now.isoformat()
                    },
                    "requests": {
                        "total": total_requests,
                        "successful": successful_requests,
                        "failed": failed_requests,
                        "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0
                    },
                    "users": {
                        "active": active_users
                    },
                    "models": [
                        {
                            "model": usage.model_name,
                            "provider": usage.provider,
                            "requests": usage.request_count,
                            "success_rate": (usage.success_count / usage.request_count * 100) if usage.request_count > 0 else 0,
                            "avg_response_time": usage.avg_response_time,
                            "avg_confidence": usage.avg_confidence
                        }
                        for usage in model_usage
                    ],
                    "languages": [
                        {
                            "language": lang[0],
                            "count": lang[1]
                        }
                        for lang in language_dist
                    ],
                    "hourly_distribution": hourly_dist
                }
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get system analytics: {str(e)}")
            return {"error": str(e)}

# ============================================================================
# EXAMPLE PLUGINS
# ============================================================================

class ExampleSecurityPlugin:
    """Example security plugin for code analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "Example Security Plugin"
        self.version = "1.0.0"
    
    async def scan_for_vulnerabilities(self, code: str, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        
        vulnerabilities = []
        
        # Simple pattern matching for demonstration
        if language.lower() == "python":
            # Check for eval usage
            if "eval(" in code:
                vulnerabilities.append({
                    "type": "Use of eval()",
                    "severity": "high",
                    "description": "The eval() function can execute arbitrary code and is dangerous with untrusted input.",
                    "line": self._find_line_number(code, "eval("),
                    "recommendation": "Use safer alternatives like ast.literal_eval() or explicit parsing."
                })
            
            # Check for pickle usage
            if "pickle.load" in code:
                vulnerabilities.append({
                    "type": "Unsafe deserialization",
                    "severity": "critical",
                    "description": "Pickle can execute arbitrary code during deserialization.",
                    "line": self._find_line_number(code, "pickle.load"),
                    "recommendation": "Use JSON or other safe serialization formats."
                })
        
        return {
            "vulnerabilities_found": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "security_score": max(0, 100 - len(vulnerabilities) * 20)
        }
    
    def _find_line_number(self, code: str, pattern: str) -> int:
        """Find line number of pattern in code"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                return i + 1
        return 0

class ExampleOptimizationPlugin:
    """Example optimization plugin for code improvement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "Example Optimization Plugin"
        self.version = "1.0.0"
    
    async def optimize_performance(self, code: str, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code for performance"""
        
        optimizations = []
        
        if language.lower() == "python":
            # Check for range(len()) pattern
            import re
            pattern = r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(\s*\w+\s*\)\s*\)'
            matches = re.finditer(pattern, code)
            
            for match in matches:
                optimizations.append({
                    "type": "Use enumerate() instead of range(len())",
                    "description": "enumerate() is more readable and efficient for getting both index and value.",
                    "line": code[:match.start()].count('\n') + 1,
                    "example_before": match.group(),
                    "example_after": "for i, item in enumerate(items):"
                })
        
        return {
            "optimizations_found": len(optimizations),
            "optimizations": optimizations,
            "estimated_improvement": f"{len(optimizations) * 5}% performance improvement"
        }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Initialize API
    api = AICodeEngineAPI(config_path)
    
    # Return FastAPI app
    return api.app

# For running directly
if __name__ == "__main__":
    import uvicorn
    
    # Create app
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

# ============================================================================
# PART 2 COMPLETE
# Next part will include:
# 1. Frontend integration examples (React components)
# 2. Advanced deployment configurations (Docker, Kubernetes)
# 3. CI/CD pipeline examples
# 4. Advanced monitoring with Grafana dashboards
# 5. Load testing and performance optimization
# 6. Security hardening and penetration testing
# 7. Multi-tenant support and scaling strategies
# ============================================================================
```