# File: advanced_ai_code_engine_part3.py
# PART 3/3: Frontend Integration, Deployment, Monitoring, and Advanced Features

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
import secrets
from pathlib import Path

#
# tertiary.py - FIX THESE IMPORTS:

# WRONG: (trying to import from non-existent modules)
# from advanced_ai_code_engine_part2 import (...)
# from advanced_ai_code_engine_part3 import (...)

# CORRECT: (import from your actual files)
from main import (
    AICodeEngine, EngineConfig, CodeGenerationRequest, CodeGenerationResponse,
    CodeLanguage, ModelProvider, TaskComplexity
)

from secondary import (
    DatabaseManager, CacheManager, RateLimiter, AuthManager,
    MetricsCollector, PluginManager, WebSocketManager, AICodeEngineAPI
)

# Then continue with your tertiary.py code...
# ============================================================================
# FRONTEND INTEGRATION - REACT COMPONENTS
# ============================================================================

class ReactComponentGenerator:
    """Generates React components for frontend integration"""
    
    def __init__(self):
        self.templates = {
            "code_editor": self._generate_code_editor_component,
            "model_selector": self._generate_model_selector_component,
            "teaching_panel": self._generate_teaching_panel_component,
            "security_scanner": self._generate_security_scanner_component,
            "performance_monitor": self._generate_performance_monitor_component
        }
    
    def generate_component(self, component_type: str, props: Dict[str, Any]) -> str:
        """Generate React component code"""
        generator = self.templates.get(component_type)
        if not generator:
            raise ValueError(f"Unknown component type: {component_type}")
        
        return generator(props)
    
    def _generate_code_editor_component(self, props: Dict[str, Any]) -> str:
        """Generate Code Editor React component"""
        return f'''
import React, {{ useState, useEffect, useCallback }} from 'react';
import {{ Box, Button, TextField, Select, MenuItem, FormControl, InputLabel, 
         CircularProgress, Alert, Snackbar, Chip, Paper, Typography }} from '@mui/material';
import {{ CodeBlock, CopyBlock, dracula }} from 'react-code-blocks';
import {{ styled }} from '@mui/material/styles';

const {props.get('componentName', 'AICodeEditor')} = ({{
  apiUrl = '{props.get('apiUrl', 'http://localhost:8000')}',
  apiKey = '{props.get('apiKey', '')}',
  defaultLanguage = '{props.get('defaultLanguage', 'python')}',
  onCodeGenerated = null,
  enableTeaching = {props.get('enableTeaching', True)},
  enableSecurityScan = {props.get('enableSecurityScan', True)},
  enablePerformanceOptimization = {props.get('enablePerformanceOptimization', False)},
}}) => {{
  // State
  const [prompt, setPrompt] = useState('');
  const [generatedCode, setGeneratedCode] = useState('');
  const [language, setLanguage] = useState(defaultLanguage);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [modelProvider, setModelProvider] = useState('ensemble');
  const [complexity, setComplexity] = useState('medium');
  const [includeExplanations, setIncludeExplanations] = useState(true);
  const [includeTests, setIncludeTests] = useState(false);
  const [securityResults, setSecurityResults] = useState(null);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [teachingResources, setTeachingResources] = useState([]);
  const [executionTime, setExecutionTime] = useState(0);
  const [confidence, setConfidence] = useState(0);

  // Available languages
  const languages = [
    {{ value: 'python', label: 'Python' }},
    {{ value: 'javascript', label: 'JavaScript' }},
    {{ value: 'typescript', label: 'TypeScript' }},
    {{ value: 'java', label: 'Java' }},
    {{ value: 'cpp', label: 'C++' }},
    {{ value: 'go', label: 'Go' }},
    {{ value: 'rust', label: 'Rust' }},
    {{ value: 'swift', label: 'Swift' }},
  ];

  // Model providers
  const modelProviders = [
    {{ value: 'ensemble', label: 'Auto-Select Best (Ensemble)' }},
    {{ value: 'deepseek-r1', label: 'DeepSeek R1 (Reasoning)' }},
    {{ value: 'gemini', label: 'Google Gemini' }},
    {{ value: 'claude', label: 'Anthropic Claude' }},
  ];

  // Complexity levels
  const complexityLevels = [
    {{ value: 'simple', label: 'Simple' }},
    {{ value: 'medium', label: 'Medium' }},
    {{ value: 'complex', label: 'Complex' }},
    {{ value: 'expert', label: 'Expert' }},
  ];

  // Generate code function
  const generateCode = async () => {{
    if (!prompt.trim()) {{
      setError('Please enter a prompt');
      return;
    }}

    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    setSecurityResults(null);
    setPerformanceMetrics(null);
    setTeachingResources([]);

    try {{
      const requestBody = {{
        prompt,
        language,
        model_provider: modelProvider === 'ensemble' ? null : modelProvider,
        complexity,
        include_explanations: includeExplanations,
        include_tests: includeTests,
        security_scan: enableSecurityScan,
        performance_optimize: enablePerformanceOptimization,
      }};

      const response = await fetch(`${{apiUrl}}/generate`, {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${{apiKey}}`,
        }},
        body: JSON.stringify(requestBody),
      }});

      const data = await response.json();

      if (!response.ok) {{
        throw new Error(data.detail || 'Generation failed');
      }}

      setGeneratedCode(data.code);
      setExecutionTime(data.execution_time);
      setConfidence(data.confidence_score);
      
      if (data.security_scan_results) {{
        setSecurityResults(data.security_scan_results);
      }}
      
      if (data.performance_metrics) {{
        setPerformanceMetrics(data.performance_metrics);
      }}
      
      if (data.teaching_resources) {{
        setTeachingResources(data.teaching_resources);
      }}

      setSuccessMessage(`Code generated successfully in ${{data.execution_time.toFixed(2)}}s`);
      
      // Callback
      if (onCodeGenerated) {{
        onCodeGenerated({{
          code: data.code,
          metadata: {{
            modelUsed: data.model_used,
            provider: data.provider,
            executionTime: data.execution_time,
            confidence: data.confidence_score,
          }},
        }});
      }}

    }} catch (err) {{
      setError(err.message);
    }} finally {{
      setIsLoading(false);
    }}
  }};

  // Copy to clipboard
  const copyToClipboard = () => {{
    navigator.clipboard.writeText(generatedCode);
    setSuccessMessage('Code copied to clipboard!');
  }};

  // Download code
  const downloadCode = () => {{
    const blob = new Blob([generatedCode], {{ type: 'text/plain' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated_code.${{language}}`;
    a.click();
    URL.revokeObjectURL(url);
  }};

  // Handle key press (Ctrl+Enter to generate)
  const handleKeyPress = (e) => {{
    if (e.ctrlKey && e.key === 'Enter') {{
      generateCode();
    }}
  }};

  // Styled components
  const EditorContainer = styled(Paper)(({ theme }) => ({{
    padding: theme.spacing(3),
    marginBottom: theme.spacing(3),
    backgroundColor: theme.palette.background.paper,
  }}));

  const CodeContainer = styled(Paper)(({ theme }) => ({{
    padding: theme.spacing(2),
    backgroundColor: '#282a36',
    color: '#f8f8f2',
    fontFamily: 'Monaco, monospace',
    fontSize: '14px',
    overflow: 'auto',
    maxHeight: '500px',
    position: 'relative',
  }}));

  const StatsContainer = styled(Box)(({ theme }) => ({{
    display: 'flex',
    gap: theme.spacing(2),
    marginTop: theme.spacing(2),
    flexWrap: 'wrap',
  }}));

  const StatChip = styled(Chip)(({ theme, severity }) => ({{
    backgroundColor: severity === 'high' ? theme.palette.error.main :
                     severity === 'medium' ? theme.palette.warning.main :
                     severity === 'low' ? theme.palette.info.main :
                     theme.palette.success.main,
    color: 'white',
    fontWeight: 'bold',
  }}));

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto', padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        AI Code Generator
      </Typography>

      {/* Input Section */}
      <EditorContainer elevation={3}>
        <Typography variant="h6" gutterBottom>
          Describe what you want to code:
        </Typography>
        
        <TextField
          fullWidth
          multiline
          rows={4}
          variant="outlined"
          placeholder="E.g., Create a Python function that calculates the Fibonacci sequence..."
          value={{prompt}}
          onChange={{ (e) => setPrompt(e.target.value) }}
          onKeyPress={{handleKeyPress}}
          disabled={{isLoading}}
          sx={{ marginBottom: 2 }}
        />

        {/* Configuration Row */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 2 }}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Language</InputLabel>
            <Select
              value={{language}}
              label="Language"
              onChange={{ (e) => setLanguage(e.target.value) }}
              disabled={{isLoading}}
            >
              {{languages.map((lang) => (
                <MenuItem key={{lang.value}} value={{lang.value}}>
                  {{lang.label}}
                </MenuItem>
              ))}}
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 180 }}>
            <InputLabel>Model Provider</InputLabel>
            <Select
              value={{modelProvider}}
              label="Model Provider"
              onChange={{ (e) => setModelProvider(e.target.value) }}
              disabled={{isLoading}}
            >
              {{modelProviders.map((model) => (
                <MenuItem key={{model.value}} value={{model.value}}>
                  {{model.label}}
                </MenuItem>
              ))}}
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Complexity</InputLabel>
            <Select
              value={{complexity}}
              label="Complexity"
              onChange={{ (e) => setComplexity(e.target.value) }}
              disabled={{isLoading}}
            >
              {{complexityLevels.map((level) => (
                <MenuItem key={{level.value}} value={{level.value}}>
                  {{level.label}}
                </MenuItem>
              ))}}
            </Select>
          </FormControl>
        </Box>

        {/* Options Row */}
        <Box sx={{ display: 'flex', gap: 2, marginBottom: 3 }}>
          <FormControlLabel
            control={{
              <Checkbox
                checked={{includeExplanations}}
                onChange={{ (e) => setIncludeExplanations(e.target.checked) }}
                disabled={{isLoading}}
              />
            }}
            label="Include Explanations"
          />
          <FormControlLabel
            control={{
              <Checkbox
                checked={{includeTests}}
                onChange={{ (e) => setIncludeTests(e.target.checked) }}
                disabled={{isLoading}}
              />
            }}
            label="Include Tests"
          />
        </Box>

        {/* Generate Button */}
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            onClick={{generateCode}}
            disabled={{isLoading || !prompt.trim()}}
            startIcon={{isLoading ? <CircularProgress size={20} /> : <PlayArrowIcon />}}
            sx={{ flexGrow: 1 }}
          >
            {{isLoading ? 'Generating...' : 'Generate Code (Ctrl+Enter)'}}
          </Button>
          
          <Button
            variant="outlined"
            onClick={{ () => setPrompt('') }}
            disabled={{isLoading}}
          >
            Clear
          </Button>
        </Box>
      </EditorContainer>

      {/* Generated Code Section */}
      {{generatedCode && (
        <EditorContainer elevation={3}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
            <Typography variant="h6">
              Generated Code ({{language.toUpperCase()}})
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={{copyToClipboard}}
                startIcon={<ContentCopyIcon />}
              >
                Copy
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={{downloadCode}}
                start-icon={<DownloadIcon />}
              >
                Download
              </Button>
            </Box>
          </Box>

          <CodeContainer>
            <CopyBlock
              text={{generatedCode}}
              language={{language}}
              showLineNumbers={{true}}
              theme={{dracula}}
              codeBlock={{true}}
            />
          </CodeContainer>

          {/* Statistics */}
          <StatsContainer>
            <StatChip
              label={`Time: ${{executionTime.toFixed(2)}}s`}
              severity="info"
            />
            <StatChip
              label={`Confidence: ${{(confidence * 100).toFixed(1)}}%`}
              severity={{confidence > 0.8 ? 'low' : confidence > 0.6 ? 'medium' : 'high'}}
            />
            {{securityResults && (
              <StatChip
                label={`Security: ${{securityResults.security_score}}/100`}
                severity={{
                  securityResults.security_score >= 90 ? 'low' :
                  securityResults.security_score >= 70 ? 'medium' : 'high'
                }}
              />
            )}}
          </StatsContainer>
        </EditorContainer>
      )}}

      {/* Security Scan Results */}
      {{securityResults && securityResults.vulnerabilities_found > 0 && (
        <EditorContainer elevation={3}>
          <Typography variant="h6" color="error" gutterBottom>
            Security Vulnerabilities Found: {{securityResults.vulnerabilities_found}}
          </Typography>
          <Box sx={{ maxHeight: '300px', overflow: 'auto' }}>
            {{securityResults.vulnerabilities.map((vuln, index) => (
              <Alert
                key={{index}}
                severity={{
                  vuln.severity === 'critical' ? 'error' :
                  vuln.severity === 'high' ? 'warning' : 'info'
                }}
                sx={{ marginBottom: 1 }}
              >
                <Typography variant="subtitle2">
                  Line {{vuln.line}}: {{vuln.type}}
                </Typography>
                <Typography variant="body2">
                  {{vuln.mitigation}}
                </Typography>
              </Alert>
            ))}}
          </Box>
        </EditorContainer>
      )}}

      {/* Teaching Resources */}
      {{enableTeaching && teachingResources.length > 0 && (
        <EditorContainer elevation={3}>
          <Typography variant="h6" gutterBottom>
            Learning Resources
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {{teachingResources.slice(0, 3).map((resource, index) => (
              <Paper key={{index}} elevation={1} sx={{ padding: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  {{resource.title}}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {{resource.content.substring(0, 200)}}{{resource.content.length > 200 ? '...' : ''}}
                </Typography>
                {{resource.code_example && (
                  <Box sx={{ marginTop: 1, backgroundColor: '#f5f5f5', padding: 1, borderRadius: 1 }}>
                    <Typography variant="caption" fontFamily="monospace">
                      {{resource.code_example}}
                    </Typography>
                  </Box>
                )}}
              </Paper>
            ))}}
          </Box>
        </EditorContainer>
      )}}

      {/* Notifications */}
      <Snackbar
        open={{!!error}}
        autoHideDuration={{6000}}
        onClose={{ () => setError(null) }}
      >
        <Alert severity="error" onClose={{ () => setError(null) }}>
          {{error}}
        </Alert>
      </Snackbar>

      <Snackbar
        open={{!!successMessage}}
        autoHideDuration={{3000}}
        onClose={{ () => setSuccessMessage(null) }}
      >
        <Alert severity="success" onClose={{ () => setSuccessMessage(null) }}>
          {{successMessage}}
        </Alert>
      </Snackbar>
    </Box>
  );
}};

export default {props.get('componentName', 'AICodeEditor')};
'''
    
    def _generate_model_selector_component(self, props: Dict[str, Any]) -> str:
        """Generate Model Selector React component"""
        return f'''
import React, {{ useState, useEffect }} from 'react';
import {{ 
  Box, Card, CardContent, Typography, Grid, Chip, 
  LinearProgress, Tooltip, IconButton 
}} from '@mui/material';
import {{ styled }} from '@mui/material/styles';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const {props.get('componentName', 'ModelSelector')} = ({{
  apiUrl = '{props.get('apiUrl', 'http://localhost:8000')}',
  onModelSelect = null,
  showDetails = {props.get('showDetails', True)},
}}) => {{
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch model performance data
  useEffect(() => {{
    fetchModels();
    // Refresh every 30 seconds
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }}, []);

  const fetchModels = async () => {{
    try {{
      const response = await fetch(`${{apiUrl}}/system/status`);
      const data = await response.json();
      
      if (data.engine?.performance_metrics) {{
        const modelData = Object.entries(data.engine.performance_metrics).map(([key, metrics]) => ({{
          id: key,
          name: key.charAt(0).toUpperCase() + key.slice(1),
          provider: metrics.provider || 'unknown',
          successRate: metrics.success_rate * 100,
          avgResponseTime: metrics.avg_response_time,
          avgConfidence: metrics.avg_confidence * 100,
          callCount: metrics.call_count,
          lastUsed: new Date(metrics.last_used).toLocaleTimeString(),
        }}));
        
        // Add ensemble option
        modelData.unshift({{
          id: 'ensemble',
          name: 'Smart Ensemble',
          provider: 'Auto-Select',
          successRate: 95, // Estimated
          avgResponseTime: 1.5,
          avgConfidence: 88,
          callCount: 0,
          lastUsed: 'Now',
          description: 'Automatically selects the best model for each task',
        }});
        
        setModels(modelData);
      }}
    }} catch (err) {{
      setError(err.message);
    }} finally {{
      setLoading(false);
    }}
  }};

  const handleModelSelect = (modelId) => {{
    setSelectedModel(modelId);
    if (onModelSelect) {{
      onModelSelect(modelId);
    }}
  }};

  // Styled components
  const ModelCard = styled(Card)(({ theme, selected }) => ({{
    cursor: 'pointer',
    border: selected ? `2px solid ${{theme.palette.primary.main}}` : '2px solid transparent',
    transition: 'all 0.2s ease-in-out',
    '&:hover': {{
      transform: 'translateY(-2px)',
      boxShadow: theme.shadows[4],
    }},
  }}));

  const PerformanceBar = styled(Box)(({ theme, value, type }) => ({{
    width: '100%',
    height: '8px',
    backgroundColor: theme.palette.grey[200],
    borderRadius: '4px',
    overflow: 'hidden',
    position: 'relative',
    '&::after': {{
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      height: '100%',
      width: `${{value}}%`,
      backgroundColor: type === 'success' ? theme.palette.success.main : 
                       type === 'confidence' ? theme.palette.info.main :
                       theme.palette.warning.main,
      transition: 'width 0.3s ease',
    }},
  }}));

  if (loading) return (
    <Box display="flex" justifyContent="center" p={3}>
      <CircularProgress />
    </Box>
  );

  if (error) return (
    <Alert severity="error">
      Failed to load models: {{error}}
    </Alert>
  );

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Select AI Model
      </Typography>
      
      <Grid container spacing={2}>
        {{models.map((model) => (
          <Grid item xs={{12}} sm={{6}} md={{4}} key={{model.id}}>
            <ModelCard
              selected={{selectedModel === model.id}}
              onClick={{ () => handleModelSelect(model.id) }}
            >
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {{model.name}}
                    </Typography>
                    <Chip
                      label={{model.provider}}
                      size="small"
                      sx={{ marginTop: 0.5 }}
                    />
                  </Box>
                  
                  {{selectedModel === model.id && (
                    <CheckCircleIcon color="primary" />
                  )}}
                </Box>

                {{showDetails && (
                  <Box mt={2}>
                    {/* Success Rate */}
                    <Box mb={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Success Rate
                        </Typography>
                        <Typography variant="caption" fontWeight="bold">
                          {{model.successRate.toFixed(1)}}%
                        </Typography>
                      </Box>
                      <PerformanceBar value={{model.successRate}} type="success" />
                    </Box>

                    {/* Confidence */}
                    <Box mb={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Avg. Confidence
                        </Typography>
                        <Typography variant="caption" fontWeight="bold">
                          {{model.avgConfidence.toFixed(1)}}%
                        </Typography>
                      </Box>
                      <PerformanceBar value={{model.avgConfidence}} type="confidence" />
                    </Box>

                    {/* Response Time */}
                    <Box mb={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Response Time
                        </Typography>
                        <Typography variant="caption" fontWeight="bold">
                          {{model.avgResponseTime.toFixed(2)}}s
                        </Typography>
                      </Box>
                      <PerformanceBar 
                        value={{Math.max(0, 100 - model.avgResponseTime * 20)}} 
                        type="response" 
                      />
                    </Box>

                    {/* Stats */}
                    <Box display="flex" justifyContent="space-between" mt={2}>
                      <Tooltip title="Total requests">
                        <Chip
                          label={`${{model.callCount}} req`}
                          size="small"
                          variant="outlined"
                        />
                      </Tooltip>
                      <Tooltip title="Last used">
                        <Chip
                          label={{model.lastUsed}}
                          size="small"
                          variant="outlined"
                        />
                      </Tooltip>
                    </Box>

                    {{model.description && (
                      <Typography variant="caption" color="text.secondary" mt={1} display="block">
                        {{model.description}}
                      </Typography>
                    )}}
                  </Box>
                )}}
              </CardContent>
            </ModelCard>
          </Grid>
        ))}}
      </Grid>
    </Box>
  );
}};

export default {props.get('componentName', 'ModelSelector')};
'''
    
    def _generate_teaching_panel_component(self, props: Dict[str, Any]) -> str:
        """Generate Teaching Panel React component"""
        return '''
// TeachingPanel.jsx - Interactive teaching component
import React, { useState, useEffect } from 'react';
import {
  Box, Accordion, AccordionSummary, AccordionDetails,
  Typography, Chip, Button, TextField, LinearProgress,
  IconButton, Card, CardContent
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import CodeIcon from '@mui/icons-material/Code';
import QuizIcon from '@mui/icons-material/Quiz';

const TeachingPanel = ({
  code = '',
  language = 'python',
  concept = '',
  difficulty = 'beginner',
  apiUrl = 'http://localhost:8000',
}) => {
  const [lessons, setLessons] = useState([]);
  const [activeLesson, setActiveLesson] = useState(null);
  const [userCode, setUserCode] = useState(code);
  const [testResults, setTestResults] = useState(null);
  const [isTesting, setIsTesting] = useState(false);
  const [explanation, setExplanation] = useState('');

  // Load lessons based on code analysis
  useEffect(() => {
    if (code) {
      analyzeCodeAndGenerateLessons();
    }
  }, [code, language]);

  const analyzeCodeAndGenerateLessons = async () => {
    try {
      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          language,
          analysis_type: 'educational'
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Generate lessons from analysis
        const generatedLessons = generateLessonsFromAnalysis(data, code, language);
        setLessons(generatedLessons);
        
        if (generatedLessons.length > 0) {
          setActiveLesson(generatedLessons[0]);
        }
      }
    } catch (error) {
      console.error('Failed to analyze code:', error);
    }
  };

  const generateLessonsFromAnalysis = (analysis, code, language) => {
    const lessons = [];
    
    // Lesson 1: Code structure
    lessons.push({
      id: 'structure',
      title: 'Code Structure & Organization',
      description: 'Learn how the code is organized and structured',
      difficulty: 'beginner',
      estimatedTime: '5 minutes',
      sections: [
        {
          type: 'explanation',
          title: 'Overall Structure',
          content: 'This code follows a modular structure with clear separation of concerns...',
        },
        {
          type: 'interactive',
          title: 'Identify Components',
          content: 'Try to identify the main components of the code below:',
          codeSnippet: code.substring(0, 200) + '...',
          questions: [
            {
              question: 'How many functions are defined?',
              answer: '3',
              hint: 'Look for "def" keyword',
            },
          ],
        },
      ],
    });

    // Lesson 2: Key concepts
    lessons.push({
      id: 'concepts',
      title: 'Key Programming Concepts',
      description: 'Learn the main concepts used in this code',
      difficulty: 'intermediate',
      estimatedTime: '10 minutes',
      sections: [
        {
          type: 'video',
          title: 'Understanding Functions',
          content: 'Watch this video to understand how functions work:',
          videoUrl: 'https://example.com/video-functions',
          duration: '3:45',
        },
        {
          type: 'quiz',
          title: 'Concept Check',
          content: 'Test your understanding of the concepts:',
          questions: [
            {
              question: 'What is the purpose of the main function?',
              options: ['Initialize variables', 'Control program flow', 'Handle errors'],
              correctAnswer: 1,
            },
          ],
        },
      ],
    });

    return lessons;
  };

  const runTests = async () => {
    setIsTesting(true);
    
    // Simulate test execution
    setTimeout(() => {
      setTestResults({
        passed: 3,
        failed: 1,
        total: 4,
        details: [
          { test: 'Test 1: Function returns correct value', passed: true },
          { test: 'Test 2: Handles edge cases', passed: true },
          { test: 'Test 3: Error handling', passed: true },
          { test: 'Test 4: Performance benchmark', passed: false },
        ],
      });
      setIsTesting(false);
    }, 2000);
  };

  const getExplanation = async (concept) => {
    try {
      const response = await fetch(`${apiUrl}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          concept,
          language,
          detail_level: 'detailed',
        }),
      });
      
      const data = await response.json();
      setExplanation(data.explanation || 'No explanation available');
    } catch (error) {
      console.error('Failed to get explanation:', error);
    }
  };

  return (
    <Box sx={{ maxWidth: '800px', margin: '0 auto' }}>
      {/* Lesson Navigation */}
      <Box display="flex" gap={1} mb={3} flexWrap="wrap">
        {lessons.map((lesson) => (
          <Chip
            key={lesson.id}
            label={lesson.title}
            onClick={() => setActiveLesson(lesson)}
            color={activeLesson?.id === lesson.id ? 'primary' : 'default'}
            variant={activeLesson?.id === lesson.id ? 'filled' : 'outlined'}
          />
        ))}
      </Box>

      {/* Active Lesson */}
      {activeLesson && (
        <Card elevation={3}>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              {activeLesson.title}
            </Typography>
            
            <Typography color="text.secondary" paragraph>
              {activeLesson.description}
            </Typography>

            <Box display="flex" gap={1} mb={2}>
              <Chip label={activeLesson.difficulty} size="small" />
              <Chip 
                label={`⏱️ ${activeLesson.estimatedTime}`} 
                size="small" 
                variant="outlined" 
              />
            </Box>

            {/* Lesson Sections */}
            {activeLesson.sections?.map((section, index) => (
              <Accordion key={index} defaultExpanded={index === 0}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {section.type === 'video' && <PlayCircleIcon />}
                    {section.type === 'quiz' && <QuizIcon />}
                    {section.type === 'interactive' && <CodeIcon />}
                    <Typography>{section.title}</Typography>
                  </Box>
                </AccordionSummary>
                
                <AccordionDetails>
                  <Typography paragraph>{section.content}</Typography>
                  
                  {section.codeSnippet && (
                    <Box
                      sx={{
                        backgroundColor: '#f5f5f5',
                        padding: 2,
                        borderRadius: 1,
                        fontFamily: 'monospace',
                        fontSize: '14px',
                        marginBottom: 2,
                      }}
                    >
                      {section.codeSnippet}
                    </Box>
                  )}

                  {section.questions?.map((q, qIndex) => (
                    <Box key={qIndex} mb={2}>
                      <Typography variant="subtitle2">
                        Question {qIndex + 1}: {q.question}
                      </Typography>
                      
                      {q.options ? (
                        <Box display="flex" flexDirection="column" gap={1} mt={1}>
                          {q.options.map((option, optIndex) => (
                            <Button
                              key={optIndex}
                              variant="outlined"
                              size="small"
                              onClick={() => {
                                // Handle answer selection
                              }}
                            >
                              {option}
                            </Button>
                          ))}
                        </Box>
                      ) : q.hint && (
                        <Typography variant="caption" color="text.secondary">
                          Hint: {q.hint}
                        </Typography>
                      )}
                    </Box>
                  ))}

                  {section.videoUrl && (
                    <Box
                      sx={{
                        width: '100%',
                        height: '200px',
                        backgroundColor: '#000',
                        borderRadius: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                      }}
                    >
                      Video: {section.videoUrl}
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>
            ))}

            {/* Interactive Code Editor */}
            <Box mt={3}>
              <Typography variant="h6" gutterBottom>
                Try It Yourself
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={6}
                value={userCode}
                onChange={(e) => setUserCode(e.target.value)}
                placeholder="Modify the code here..."
                sx={{
                  fontFamily: 'monospace',
                  '& .MuiInputBase-input': {
                    fontFamily: 'monospace',
                  },
                }}
              />
              
              <Box display="flex" gap={2} mt={2}>
                <Button
                  variant="contained"
                  onClick={runTests}
                  disabled={isTesting}
                  startIcon={<PlayCircleIcon />}
                >
                  {isTesting ? 'Running Tests...' : 'Run Tests'}
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={() => getExplanation(activeLesson.title)}
                >
                  Get Detailed Explanation
                </Button>
              </Box>

              {/* Test Results */}
              {testResults && (
                <Box mt={2}>
                  <Typography variant="subtitle1">
                    Test Results: {testResults.passed}/{testResults.total} passed
                  </Typography>
                  
                  <LinearProgress
                    variant="determinate"
                    value={(testResults.passed / testResults.total) * 100}
                    sx={{ mt: 1, mb: 2 }}
                  />
                  
                  {testResults.details.map((test, index) => (
                    <Box
                      key={index}
                      display="flex"
                      alignItems="center"
                      gap={1}
                      mb={1}
                    >
                      {test.passed ? '✅' : '❌'}
                      <Typography variant="body2">{test.test}</Typography>
                    </Box>
                  ))}
                </Box>
              )}

              {/* Explanation */}
              {explanation && (
                <Box mt={2} p={2} bgcolor="info.light" borderRadius={1}>
                  <Typography variant="subtitle2" gutterBottom>
                    Explanation:
                  </Typography>
                  <Typography variant="body2">{explanation}</Typography>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default TeachingPanel;
'''
    
    def _generate_security_scanner_component(self, props: Dict[str, Any]) -> str:
        """Generate Security Scanner React component"""
        return '''
// SecurityScanner.jsx - Interactive security analysis component
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, LinearProgress,
  List, ListItem, ListItemIcon, ListItemText,
  Chip, Button, Alert, AlertTitle, Dialog,
  DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';

const SecurityScanner = ({
  code = '',
  language = 'python',
  apiUrl = 'http://localhost:8000',
  autoScan = true,
  onScanComplete = null,
}) => {
  const [scanResults, setScanResults] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [selectedVulnerability, setSelectedVulnerability] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);

  // Auto-scan on code change
  useEffect(() => {
    if (autoScan && code.trim()) {
      scanCode();
    }
  }, [code, language]);

  const scanCode = async () => {
    setIsScanning(true);
    
    try {
      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          language,
          analysis_type: 'security',
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        const securityData = data.security_analysis || {
          vulnerabilities_found: 0,
          security_score: 100,
          vulnerabilities: [],
        };
        
        setScanResults(securityData);
        
        // Add to history
        setScanHistory(prev => [{
          timestamp: new Date().toLocaleTimeString(),
          score: securityData.security_score,
          vulnerabilities: securityData.vulnerabilities_found,
          codeSnippet: code.substring(0, 100) + '...',
        }, ...prev.slice(0, 9)]);
        
        if (onScanComplete) {
          onScanComplete(securityData);
        }
      }
    } catch (error) {
      console.error('Security scan failed:', error);
    } finally {
      setIsScanning(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'medium': return '#ffb300';
      case 'low': return '#1976d2';
      default: return '#757575';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity.toLowerCase()) {
      case 'critical': return <ErrorIcon />;
      case 'high': return <WarningIcon />;
      case 'medium': return <WarningIcon />;
      case 'low': return <InfoIcon />;
      default: return <InfoIcon />;
    }
  };

  const SecurityScoreCard = ({ score }) => {
    let color;
    let level;
    
    if (score >= 90) {
      color = '#4caf50';
      level = 'Excellent';
    } else if (score >= 70) {
      color = '#ff9800';
      level = 'Good';
    } else if (score >= 50) {
      color = '#ff5722';
      level = 'Fair';
    } else {
      color = '#f44336';
      level = 'Poor';
    }
    
    return (
      <Card sx={{ backgroundColor: color + '15', border: `1px solid ${color}30` }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box>
              <Typography variant="h3" color={color}>
                {score}
              </Typography>
              <Typography color="text.secondary">Security Score</Typography>
            </Box>
            <Box textAlign="right">
              <Chip label={level} sx={{ backgroundColor: color, color: 'white' }} />
              <Typography variant="caption" display="block" mt={1}>
                0 = Worst, 100 = Best
              </Typography>
            </Box>
          </Box>
          
          <LinearProgress
            variant="determinate"
            value={score}
            sx={{
              mt: 2,
              height: 8,
              borderRadius: 4,
              backgroundColor: color + '30',
              '& .MuiLinearProgress-bar': {
                backgroundColor: color,
              },
            }}
          />
        </CardContent>
      </Card>
    );
  };

  const VulnerabilityDetails = ({ vulnerability }) => (
    <Dialog
      open={!!vulnerability}
      onClose={() => setSelectedVulnerability(null)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          {getSeverityIcon(vulnerability?.severity)}
          {vulnerability?.type}
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {vulnerability && (
          <>
            <Box display="flex" gap={1} mb={2}>
              <Chip
                label={`Line ${vulnerability.line}`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={vulnerability.severity}
                size="small"
                sx={{
                  backgroundColor: getSeverityColor(vulnerability.severity),
                  color: 'white',
                }}
              />
            </Box>
            
            <Typography paragraph>
              <strong>Description:</strong> {vulnerability.description}
            </Typography>
            
            <Typography paragraph>
              <strong>Risk:</strong> {vulnerability.severity} severity vulnerability
            </Typography>
            
            <Box mb={2}>
              <Typography variant="subtitle2" gutterBottom>
                Vulnerable Code:
              </Typography>
              <Box
                sx={{
                  backgroundColor: '#2d2d2d',
                  color: '#f8f8f2',
                  padding: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '14px',
                }}
              >
                {vulnerability.code_snippet}
              </Box>
            </Box>
            
            <Alert severity="info">
              <AlertTitle>Recommended Fix</AlertTitle>
              {vulnerability.mitigation}
            </Alert>
            
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>
                Prevention Tips:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>✓</ListItemIcon>
                  <ListItemText primary="Always validate user input" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>✓</ListItemIcon>
                  <ListItemText primary="Use prepared statements for database queries" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>✓</ListItemIcon>
                  <ListItemText primary="Keep dependencies updated" />
                </ListItem>
              </List>
            </Box>
          </>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={() => setSelectedVulnerability(null)}>
          Close
        </Button>
        <Button
          variant="contained"
          onClick={() => {
            // Implement auto-fix
            setSelectedVulnerability(null);
          }}
        >
          Apply Auto-Fix
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">
          <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Security Analysis
        </Typography>
        
        <Button
          variant="outlined"
          onClick={scanCode}
          disabled={isScanning || !code.trim()}
          startIcon={<SecurityIcon />}
        >
          {isScanning ? 'Scanning...' : 'Scan Code'}
        </Button>
      </Box>

      {isScanning ? (
        <Box textAlign="center" py={4}>
          <LinearProgress sx={{ mb: 2 }} />
          <Typography>Analyzing code for security vulnerabilities...</Typography>
        </Box>
      ) : scanResults ? (
        <>
          {/* Security Score */}
          <Box mb={3}>
            <SecurityScoreCard score={scanResults.security_score} />
          </Box>

          {/* Vulnerabilities */}
          {scanResults.vulnerabilities_found > 0 ? (
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="error" gutterBottom>
                  ⚠️ {scanResults.vulnerabilities_found} Vulnerability(s) Found
                </Typography>
                
                <List>
                  {scanResults.vulnerabilities.map((vuln, index) => (
                    <ListItem
                      key={index}
                      button
                      onClick={() => setSelectedVulnerability(vuln)}
                      sx={{
                        borderBottom: '1px solid #eee',
                        '&:hover': {
                          backgroundColor: '#f5f5f5',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ color: getSeverityColor(vuln.severity) }}>
                        {getSeverityIcon(vuln.severity)}
                      </ListItemIcon>
                      
                      <ListItemText
                        primary={vuln.type}
                        secondary={`Line ${vuln.line}: ${vuln.description.substring(0, 100)}...`}
                      />
                      
                      <Chip
                        label={vuln.severity}
                        size="small"
                        sx={{
                          backgroundColor: getSeverityColor(vuln.severity),
                          color: 'white',
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          ) : (
            <Alert severity="success" icon={<CheckCircleIcon />}>
              <AlertTitle>No Security Issues Found</AlertTitle>
              Your code passed all security checks. Great job!
            </Alert>
          )}

          {/* Security Recommendations */}
          <Box mt={3}>
            <Typography variant="h6" gutterBottom>
              Security Recommendations
            </Typography>
            
            <Box display="flex" flexWrap="wrap" gap={2}>
              {[
                {
                  title: 'Input Validation',
                  description: 'Always validate and sanitize user input',
                  icon: '🔐',
                },
                {
                  title: 'Dependency Scanning',
                  description: 'Regularly scan dependencies for vulnerabilities',
                  icon: '📦',
                },
                {
                  title: 'Authentication',
                  description: 'Implement proper authentication and authorization',
                  icon: '👤',
                },
                {
                  title: 'Error Handling',
                  description: 'Avoid exposing sensitive information in errors',
                  icon: '⚠️',
                },
              ].map((rec, index) => (
                <Card key={index} sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <CardContent>
                    <Typography variant="h4" mb={1}>
                      {rec.icon}
                    </Typography>
                    <Typography variant="subtitle1">
                      {rec.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {rec.description}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Box>

          {/* Scan History */}
          {scanHistory.length > 0 && (
            <Box mt={3}>
              <Typography variant="h6" gutterBottom>
                Scan History
              </Typography>
              
              <List dense>
                {scanHistory.map((scan, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {scan.vulnerabilities === 0 ? '✅' : '⚠️'}
                    </ListItemIcon>
                    <ListItemText
                      primary={`Score: ${scan.score} | Vulnerabilities: ${scan.vulnerabilities}`}
                      secondary={`${scan.timestamp} - ${scan.codeSnippet}`}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          <VulnerabilityDetails vulnerability={selectedVulnerability} />
        </>
      ) : (
        <Alert severity="info">
          Enter some code and click "Scan Code" to check for security vulnerabilities.
        </Alert>
      )}
    </Box>
  );
};

export default SecurityScanner;
'''
    
    def _generate_performance_monitor_component(self, props: Dict[str, Any]) -> str:
        """Generate Performance Monitor React component"""
        return '''
// PerformanceMonitor.jsx - Real-time performance monitoring dashboard
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Grid,
  LinearProgress, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow,
  Paper, IconButton, Tooltip, Button
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TimelineIcon from '@mui/icons-material/Timeline';
import SpeedIcon from '@mui/icons-material/Speed';
import MemoryIcon from '@mui/icons-material/Memory';
import ShowChartIcon from '@mui/icons-material/ShowChart';

const PerformanceMonitor = ({
  apiUrl = 'http://localhost:8000',
  refreshInterval = 10000,
  showRealTime = true,
}) => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch metrics periodically
  useEffect(() => {
    fetchMetrics();
    
    if (showRealTime) {
      const interval = setInterval(fetchMetrics, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [refreshInterval, showRealTime]);

  const fetchMetrics = async () => {
    setLoading(true);
    
    try {
      const response = await fetch(`${apiUrl}/system/status`);
      const data = await response.json();
      
      setMetrics(data);
      
      // Add to history
      const timestamp = new Date().toLocaleTimeString();
      setHistory(prev => [{
        timestamp,
        requests: data.engine?.total_requests || 0,
        memory: Math.random() * 80 + 20, // Simulated
        cpu: Math.random() * 60 + 20, // Simulated
        responseTime: data.engine?.avg_response_time || 0,
      }, ...prev.slice(0, 20)]);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ title, value, unit, icon, color, trend = 0 }) => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography color="text.secondary" variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" mt={1}>
              {value}
              <Typography component="span" variant="h6" color="text.secondary">
                {unit}
              </Typography>
            </Typography>
          </Box>
          
          <Box>
            {icon}
            {trend !== 0 && (
              <Chip
                label={`${trend > 0 ? '+' : ''}${trend.toFixed(1)}%`}
                size="small"
                color={trend > 0 ? 'error' : 'success'}
                sx={{ ml: 1 }}
              />
            )}
          </Box>
        </Box>
        
        <LinearProgress
          variant="determinate"
          value={Math.min(100, (value / 100) * 100)}
          sx={{
            mt: 2,
            height: 6,
            borderRadius: 3,
            backgroundColor: color + '20',
            '& .MuiLinearProgress-bar': {
              backgroundColor: color,
            },
          }}
        />
      </CardContent>
    </Card>
  );

  const PerformanceChart = ({ data, title }) => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        
        <Box sx={{ height: '200px', position: 'relative' }}>
          {/* Simplified chart rendering */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              display: 'flex',
              alignItems: 'flex-end',
              height: '160px',
              gap: '2px',
            }}
          >
            {history.slice().reverse().map((point, index) => (
              <Tooltip
                key={index}
                title={`${point.timestamp}: ${point.requests} requests`}
              >
                <Box
                  sx={{
                    flex: 1,
                    height: `${(point.requests / Math.max(...history.map(h => h.requests), 1)) * 100}%`,
                    backgroundColor: '#2196f3',
                    '&:hover': {
                      backgroundColor: '#1976d2',
                    },
                  }}
                />
              </Tooltip>
            ))}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (!metrics) {
    return (
      <Box textAlign="center" py={4}>
        <LinearProgress sx={{ mb: 2 }} />
        <Typography>Loading performance metrics...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">
          <SpeedIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Performance Dashboard
        </Typography>
        
        <Box>
          <Tooltip title="Refresh">
            <IconButton onClick={fetchMetrics} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Button
            variant="outlined"
            size="small"
            sx={{ ml: 1 }}
            onClick={() => setActiveTab(activeTab === 'overview' ? 'details' : 'overview')}
          >
            {activeTab === 'overview' ? 'Show Details' : 'Show Overview'}
          </Button>
        </Box>
      </Box>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          {/* Key Metrics */}
          <Grid container spacing={3} mb={3}>
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Total Requests"
                value={metrics.engine?.total_requests || 0}
                unit=""
                icon={<TimelineIcon sx={{ color: '#2196f3' }} />}
                color="#2196f3"
                trend={5.2}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Avg Response Time"
                value={metrics.engine?.avg_response_time?.toFixed(2) || 0}
                unit="s"
                icon={<SpeedIcon sx={{ color: '#4caf50' }} />}
                color="#4caf50"
                trend={-2.1}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Success Rate"
                value={((metrics.engine?.success_rate || 0) * 100).toFixed(1)}
                unit="%"
                icon={<ShowChartIcon sx={{ color: '#ff9800' }} />}
                color="#ff9800"
                trend={1.5}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Active Models"
                value={metrics.engine?.models_available?.length || 0}
                unit=""
                icon={<MemoryIcon sx={{ color: '#9c27b0' }} />}
                color="#9c27b0"
              />
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={3} mb={3}>
            <Grid item xs={12} md={8}>
              <PerformanceChart
                data={history}
                title="Request Rate (Last 20 Updates)"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Performance
                  </Typography>
                  
                  {metrics.engine?.performance_metrics && Object.entries(
                    metrics.engine.performance_metrics
                  ).map(([model, data]) => (
                    <Box key={model} mb={2}>
                      <Typography variant="body2">
                        {model.charAt(0).toUpperCase() + model.slice(1)}
                      </Typography>
                      
                      <Box display="flex" alignItems="center" gap={1}>
                        <LinearProgress
                          variant="determinate"
                          value={(data.success_rate || 0) * 100}
                          sx={{
                            flexGrow: 1,
                            height: 8,
                            borderRadius: 4,
                          }}
                        />
                        <Typography variant="caption">
                          {(data.success_rate * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}

      {/* Details Tab */}
      {activeTab === 'details' && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Detailed Performance Metrics
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell align="right">Value</TableCell>
                    <TableCell align="right">Min</TableCell>
                    <TableCell align="right">Max</TableCell>
                    <TableCell align="right">Trend</TableCell>
                  </TableRow>
                </TableHead>
                
                <TableBody>
                  <TableRow>
                    <TableCell>Requests per Minute</TableCell>
                    <TableCell align="right">45.2</TableCell>
                    <TableCell align="right">12.1</TableCell>
                    <TableCell align="right">89.3</TableCell>
                    <TableCell align="right">
                      <Chip label="+5.2%" size="small" color="success" />
                    </TableCell>
                  </TableRow>
                  
                  <TableRow>
                    <TableCell>Average Response Time</TableCell>
                    <TableCell align="right">1.24s</TableCell>
                    <TableCell align="right">0.89s</TableCell>
                    <TableCell align="right">3.45s</TableCell>
                    <TableCell align="right">
                      <Chip label="-2.1%" size="small" color="error" />
                    </TableCell>
                  </TableRow>
                  
                  <TableRow>
                    <TableCell>Error Rate</TableCell>
                    <TableCell align="right">2.3%</TableCell>
                    <TableCell align="right">1.1%</TableCell>
                    <TableCell align="right">8.9%</TableCell>
                    <TableCell align="right">
                      <Chip label="-0.5%" size="small" color="success" />
                    </TableCell>
                  </TableRow>
                  
                  <TableRow>
                    <TableCell>Cache Hit Rate</TableCell>
                    <TableCell align="right">78.5%</TableCell>
                    <TableCell align="right">65.2%</TableCell>
                    <TableCell align="right">92.1%</TableCell>
                    <TableCell align="right">
                      <Chip label="+3.2%" size="small" color="success" />
                    </TableCell>
                  </TableRow>
                  
                  <TableRow>
                    <TableCell>Memory Usage</TableCell>
                    <TableCell align="right">45%</TableCell>
                    <TableCell align="right">32%</TableCell>
                    <TableCell align="right">78%</TableCell>
                    <TableCell align="right">
                      <Chip label="+1.8%" size="small" color="warning" />
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* System Health */}
      <Box mt={3}>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        
        <Grid container spacing={2}>
          {[
            { service: 'API Server', status: 'healthy', latency: '24ms' },
            { service: 'Database', status: 'healthy', latency: '12ms' },
            { service: 'Cache', status: 'degraded', latency: '45ms' },
            { service: 'AI Models', status: 'healthy', latency: '156ms' },
          ].map((service, index) => (
            <Grid item xs={12} sm={6} key={index}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="subtitle1">
                        {service.service}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Latency: {service.latency}
                      </Typography>
                    </Box>
                    
                    <Chip
                      label={service.status}
                      size="small"
                      color={
                        service.status === 'healthy' ? 'success' :
                        service.status === 'degraded' ? 'warning' : 'error'
                      }
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default PerformanceMonitor;
'''

# ============================================================================
# DEPLOYMENT CONFIGURATION - DOCKER & KUBERNETES
# ============================================================================

class DeploymentConfigGenerator:
    """Generates deployment configurations"""
    
    def generate_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate Dockerfile for the engine"""
        return f'''
# Multi-stage build for AI Code Engine
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs plugins cache

# Environment variables
ENV PYTHONPATH=/home/appuser
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${{PORT}}/health || exit 1

# Expose port
EXPOSE ${{PORT}}

# Run application
CMD ["python", "-m", "uvicorn", "advanced_ai_code_engine_part2:create_app", \
     "--host", "0.0.0.0", "--port", "${{PORT}}", \
     "--workers", "4", "--timeout-keep-alive", "30"]
'''
    
    def generate_docker_compose(self, config: Dict[str, Any]) -> str:
        """Generate docker-compose.yml for full stack"""
        return f'''
version: '3.8'

services:
  # AI Code Engine API
  ai-code-engine:
    build: .
    container_name: ai-code-engine
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ai_engine
      - REDIS_ENABLED=true
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - DEEPSEEK_API_KEY={config.get('deepseek_api_key', 'your-deepseek-key')}
      - GOOGLE_API_KEY={config.get('google_api_key', 'your-google-key')}
      - ANTHROPIC_API_KEY={config.get('anthropic_api_key', 'your-anthropic-key')}
      - OPENAI_API_KEY={config.get('openai_api_key', 'your-openai-key')}
      - AUTH_SECRET_KEY={config.get('auth_secret_key', 'change-me-to-secure-random-string')}
    volumes:
      - ./logs:/home/appuser/logs
      - ./plugins:/home/appuser/plugins
      - ./cache:/home/appuser/cache
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - ai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: ai-engine-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=ai_engine
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - ai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ai-engine-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - ai-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Frontend (React)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.frontend
    container_name: ai-engine-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - REACT_APP_WS_URL=ws://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - ai-code-engine
    networks:
      - ai-network

  # Monitoring (Prometheus + Grafana)
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - ai-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    networks:
      - ai-network

  # Load Balancer (Nginx)
  nginx:
    image: nginx:alpine
    container_name: nginx-load-balancer
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-code-engine
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
'''
    
    def generate_kubernetes_manifests(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate Kubernetes manifests"""
        
        manifests = {}
        
        # Namespace
        manifests['namespace.yaml'] = '''
apiVersion: v1
kind: Namespace
metadata:
  name: ai-code-engine
  labels:
    name: ai-code-engine
'''
        
        # ConfigMap
        manifests['configmap.yaml'] = f'''
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-engine-config
  namespace: ai-code-engine
data:
  DATABASE_URL: "postgresql://postgres:$(POSTGRES_PASSWORD)@ai-engine-db:5432/ai_engine"
  REDIS_ENABLED: "true"
  REDIS_HOST: "ai-engine-redis"
  REDIS_PORT: "6379"
  CACHE_TTL: "3600"
  ENABLE_RATE_LIMITING: "true"
  RATE_LIMIT_DEFAULT: "100"
  PLUGIN_DIR: "/app/plugins"
  CORS_ORIGINS: "*"
'''
        
        # Secrets
        manifests['secrets.yaml'] = '''
apiVersion: v1
kind: Secret
metadata:
  name: ai-engine-secrets
  namespace: ai-code-engine
type: Opaque
stringData:
  DEEPSEEK_API_KEY: "your-deepseek-api-key"
  GOOGLE_API_KEY: "your-google-api-key"
  ANTHROPIC_API_KEY: "your-anthropic-api-key"
  OPENAI_API_KEY: "your-openai-api-key"
  AUTH_SECRET_KEY: "change-me-to-secure-random-string"
  POSTGRES_PASSWORD: "secure-db-password"
'''
        
        # PostgreSQL StatefulSet
        manifests['postgresql-statefulset.yaml'] = '''
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ai-engine-db
  namespace: ai-code-engine
spec:
  serviceName: ai-engine-db
  replicas: 1
  selector:
    matchLabels:
      app: ai-engine-db
  template:
    metadata:
      labels:
        app: ai-engine-db
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "ai_engine"
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: POSTGRES_PASSWORD
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command: ["pg_isready", "-U", "postgres"]
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command: ["pg_isready", "-U", "postgres"]
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: "10Gi"
'''
        
        # Redis Deployment
        manifests['redis-deployment.yaml'] = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-engine-redis
  namespace: ai-code-engine
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-engine-redis
  template:
    metadata:
      labels:
        app: ai-engine-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server", "--appendonly", "yes", "--maxmemory", "512mb", "--maxmemory-policy", "allkeys-lru"]
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          exec:
            command: ["redis-cli", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command: ["redis-cli", "ping"]
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-data
        emptyDir: {}
'''
        
        # AI Engine Deployment
        manifests['ai-engine-deployment.yaml'] = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-code-engine
  namespace: ai-code-engine
  labels:
    app: ai-code-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-code-engine
  template:
    metadata:
      labels:
        app: ai-code-engine
    spec:
      containers:
      - name: ai-code-engine
        image: ai-code-engine:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: ai-engine-config
              key: DATABASE_URL
        - name: REDIS_ENABLED
          valueFrom:
            configMapKeyRef:
              name: ai-engine-config
              key: REDIS_ENABLED
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: ai-engine-config
              key: REDIS_HOST
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: ai-engine-config
              key: REDIS_PORT
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: DEEPSEEK_API_KEY
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: GOOGLE_API_KEY
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: ANTHROPIC_API_KEY
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: OPENAI_API_KEY
        - name: AUTH_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ai-engine-secrets
              key: AUTH_SECRET_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        volumeMounts:
        - name: plugins-volume
          mountPath: /app/plugins
        - name: cache-volume
          mountPath: /app/cache
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: plugins-volume
        emptyDir: {}
      - name: cache-volume
        emptyDir: {}
      - name: logs-volume
        emptyDir: {}
'''
        
        # Services
        manifests['services.yaml'] = '''
apiVersion: v1
kind: Service
metadata:
  name: ai-engine-db
  namespace: ai-code-engine
spec:
  selector:
    app: ai-engine-db
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: ai-engine-redis
  namespace: ai-code-engine
spec:
  selector:
    app: ai-engine-redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: ai-code-engine
  namespace: ai-code-engine
spec:
  selector:
    app: ai-code-engine
  ports:
  - port: 8000
    targetPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ai-code-engine-loadbalancer
  namespace: ai-code-engine
spec:
  type: LoadBalancer
  selector:
    app: ai-code-engine
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 443
    targetPort: 8000
    protocol: TCP
    name: https
'''
        
        # Horizontal Pod Autoscaler
        manifests['hpa.yaml'] = '''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-code-engine-hpa
  namespace: ai-code-engine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-code-engine
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
'''
        
        # Ingress
        manifests['ingress.yaml'] = '''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-code-engine-ingress
  namespace: ai-code-engine
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - ai-code-engine.example.com
    secretName: ai-engine-tls
  rules:
  - host: ai-code-engine.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-code-engine
            port:
              number: 8000
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: ai-code-engine
            port:
              number: 8000
'''
        
        return manifests
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration"""
        return '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules.yml"

scrape_configs:
  - job_name: 'ai-code-engine'
    static_configs:
      - targets: ['ai-code-engine.ai-code-engine.svc.cluster.local:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
'''

# ============================================================================
# CI/CD PIPELINE - GITHUB ACTIONS
# ============================================================================

class CICDPipelineGenerator:
    """Generates CI/CD pipeline configurations"""
    
    def generate_github_actions(self) -> str:
        """Generate GitHub Actions workflow"""
        return '''
name: AI Code Engine CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  DOCKER_BUILDKIT: 1

jobs:
  # Lint and Test
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install pytest pytest-cov pytest-asyncio black flake8 mypy
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Check formatting with black
      run: |
        black --check --diff .
    
    - name: Type check with mypy
      run: |
        mypy --install-types --non-interactive .
    
    - name: Run tests with pytest
      run: |
        pytest tests/ --cov=advanced_ai_code_engine --cov-report=xml --cov-report=html --verbose
      env:
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  # Build and Push Docker Image
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' || github.event_name == 'release'
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata for Docker
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{sha}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

  # Deploy to Kubernetes
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Set up kustomize
      uses: imranismail/setup-kustomize@v2
      with:
        kustomize-version: '5.0.0'
    
    - name: Configure Kubernetes
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBECONFIG }}" > $HOME/.kube/config
        chmod 600 $HOME/.kube/config
    
    - name: Deploy to Kubernetes
      run: |
        # Update image tag in kustomization
        cd k8s
        kustomize edit set image ai-code-engine=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        
        # Apply manifests
        kustomize build . | kubectl apply -f -
        
        # Wait for rollout
        kubectl rollout status deployment/ai-code-engine -n ai-code-engine --timeout=300s
        
        # Run smoke tests
        kubectl run smoke-test --rm -i --restart=Never --image=curlimages/curl \
          -- curl -f http://ai-code-engine.ai-code-engine.svc.cluster.local:8000/health
    
    - name: Run integration tests
      run: |
        # Install test dependencies
        pip install -r requirements-test.txt
        
        # Run integration tests
        python -m pytest tests/integration/ --verbose
        
        # Run load tests
        locust -f tests/load_test.py --headless -u 10 -r 1 --run-time 1m --host=http://ai-code-engine.ai-code-engine.svc.cluster.local:8000

  # Security Scan
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Bandit security linter
      run: |
        pip install bandit
        bandit -r . -f json -o bandit-results.json
    
    - name: Run Snyk security scan
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high

  # Performance Tests
  performance-test:
    needs: deploy
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run k6 load tests
      uses: grafana/k6-action@v0.3.0
      with:
        filename: tests/loadtest.js
    
    - name: Generate performance report
      run: |
        # Generate performance metrics
        python tests/performance_analyzer.py
        
        # Upload artifacts
        mkdir -p performance-report
        mv performance_*.csv performance-report/
        mv performance_*.html performance-report/
        
    - name: Upload performance report
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance-report/
'''

# ============================================================================
# LOAD TESTING & PERFORMANCE OPTIMIZATION
# ============================================================================

class LoadTestingGenerator:
    """Generates load testing configurations"""
    
    def generate_locust_test(self) -> str:
        """Generate Locust load test script"""
        return '''
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import random
import json

class AICodeEngineUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get token"""
        response = self.client.post("/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
        else:
            self.token = None
    
    @task(3)
    def generate_code(self):
        """Generate code with various prompts"""
        prompts = [
            "Create a Python function to calculate factorial",
            "Write a React component for a login form",
            "Implement a REST API endpoint in Node.js",
            "Create a SQL query to find duplicate records",
            "Write a Dockerfile for a Python web application",
        ]
        
        languages = ["python", "javascript", "typescript", "java", "go"]
        
        payload = {
            "prompt": random.choice(prompts),
            "language": random.choice(languages),
            "include_explanations": random.choice([True, False]),
            "security_scan": random.choice([True, False]),
        }
        
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        with self.client.post("/generate", 
                             json=payload,
                             headers=headers,
                             catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure("Generation failed")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def analyze_code(self):
        """Analyze existing code"""
        code_samples = [
            '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
''',
            '''
function validateEmail(email) {
    const re = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return re.test(email);
}
''',
            '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        ]
        
        payload = {
            "code": random.choice(code_samples),
            "language": "python",
            "analysis_type": "comprehensive"
        }
        
        self.client.post("/analyze", json=payload)
    
    @task(1)
    def get_metrics(self):
        """Get system metrics"""
        self.client.get("/system/status")
    
    @task(1)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test data"""
    if isinstance(environment.runner, MasterRunner):
        print("Initializing test data...")
        
        # Create test user
        from locust.clients import HttpSession
        client = HttpSession(base_url=environment.host)
        
        response = client.post("/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass"
        })
        
        if response.status_code == 200:
            print("Test user created successfully")
        else:
            print(f"Failed to create test user: {response.text}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting load test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")

# Custom metrics
from locust import stats
import time

class CustomStats(stats.StatsEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentiles = {}
    
    def get_current_response_time_percentile(self, percentile):
        """Get response time percentile"""
        if not self.response_times:
            return 0
        
        sorted_times = sorted(self.response_times.keys())
        total = sum(self.response_times.values())
        
        running_sum = 0
        for time_val in sorted_times:
            running_sum += self.response_times[time_val]
            if (running_sum / total) * 100 >= percentile:
                return time_val
        return sorted_times[-1]

# Replace default stats class
stats.StatsEntry = CustomStats
'''
    
    def generate_k6_test(self) -> str:
        """Generate k6 load test script"""
        return '''
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const generationSuccessRate = new Rate('generation_success_rate');
const generationTrend = new Trend('generation_trend');
const errorCounter = new Counter('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 10 },   // Stay at 10 users
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '1m', target: 10 },   // Ramp down to 10 users
    { duration: '1m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000'], // 95% of requests under 2s
    'generation_success_rate': ['rate>0.95'], // Success rate > 95%
    'errors': ['count<100'], // Less than 100 errors
  },
};

// Test data
const prompts = [
  'Create a Python function to reverse a string',
  'Write a JavaScript function to validate email',
  'Implement a binary search algorithm in Java',
  'Create a Dockerfile for a Node.js application',
  'Write a SQL query to find the top 10 customers',
];

const languages = ['python', 'javascript', 'typescript', 'java', 'go'];

// Authentication token
let authToken = '';

export function setup() {
  // Login and get token
  const loginRes = http.post(`${__ENV.BASE_URL}/auth/login`, {
    username: 'testuser',
    password: 'testpass',
  });
  
  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });
  
  return {
    token: loginRes.json('access_token'),
  };
}

export default function (data) {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${data.token}`,
    },
  };
  
  // Group related requests
  group('AI Code Engine API', function () {
    // Code generation
    const payload = {
      prompt: prompts[Math.floor(Math.random() * prompts.length)],
      language: languages[Math.floor(Math.random() * languages.length)],
      include_explanations: Math.random() > 0.5,
      security_scan: Math.random() > 0.5,
    };
    
    const genRes = http.post(
      `${__ENV.BASE_URL}/generate`,
      JSON.stringify(payload),
      params
    );
    
    // Track metrics
    const genSuccess = check(genRes, {
      'generation status 200': (r) => r.status === 200,
      'generation successful': (r) => r.json('success') === true,
    });
    
    generationSuccessRate.add(genSuccess);
    generationTrend.add(genRes.timings.duration);
    
    if (!genSuccess) {
      errorCounter.add(1);
    }
    
    // Code analysis
    const analysisPayload = {
      code: 'def hello():\n    print("Hello World")',
      language: 'python',
      analysis_type: 'comprehensive',
    };
    
    const analysisRes = http.post(
      `${__ENV.BASE_URL}/analyze`,
      JSON.stringify(analysisPayload),
      params
    );
    
    check(analysisRes, {
      'analysis status 200': (r) => r.status === 200,
    });
    
    // System status
    const statusRes = http.get(`${__ENV.BASE_URL}/system/status`, params);
    
    check(statusRes, {
      'status check 200': (r) => r.status === 200,
    });
    
    // Health check
    const healthRes = http.get(`${__ENV.BASE_URL}/health`);
    
    check(healthRes, {
      'health check 200': (r) => r.status === 200,
      'health check healthy': (r) => r.json('status') === 'healthy',
    });
  });
  
  sleep(Math.random() * 2 + 1); // Random sleep between 1-3 seconds
}

export function teardown(data) {
  // Cleanup if needed
  console.log('Test completed');
}
'''
    
    def generate_performance_optimization_guide(self) -> str:
        """Generate performance optimization guide"""
        return '''
# AI Code Engine Performance Optimization Guide

## 1. Database Optimization

### Indexing Strategy
```sql
-- Essential indexes for AI Code Engine
CREATE INDEX idx_requests_user_id ON request_history(user_id);
CREATE INDEX idx_requests_created_at ON request_history(created_at);
CREATE INDEX idx_requests_language ON request_history(language);
CREATE INDEX idx_code_analysis_hash ON code_analysis(code_hash);
CREATE INDEX idx_cache_expires ON cache_entries(expires_at);
CREATE INDEX idx_model_usage_bucket ON model_usage(hour_bucket);

-- Composite indexes for common queries
CREATE INDEX idx_user_activity ON request_history(user_id, created_at DESC);
CREATE INDEX idx_language_stats ON request_history(language, created_at, success);