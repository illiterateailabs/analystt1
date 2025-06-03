"""
Tests for the Templates API endpoints.

This module contains comprehensive tests for the templates API, covering:
- Template suggestions based on use cases
- CRUD operations for templates (create, read, update, delete)
- Template listing with pagination
- Template request creation
- Validation (name validation, process type validation)
- Authorization (admin vs analyst roles)
- File operations (save, load, delete)
- Auto-generation from use cases
- Edge cases and error handling
- Mocking of all external dependencies (file system, CrewFactory, etc.)
"""

import os
import json
import yaml
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from fastapi import status
from fastapi.testclient import TestClient
from pathlib import Path
from datetime import datetime

from backend.main import app
from backend.auth.rbac import Roles, RoleSets
from backend.agents.factory import CrewFactory
from backend.integrations.gemini_client import GeminiClient
from backend.api.v1.templates import (
    AGENT_CONFIGS_CREWS_DIR,
    get_template_path,
    template_exists,
    load_template,
    save_template,
    delete_template_file,
    template_to_response,
    generate_template_suggestions,
    generate_template_from_use_case,
    auto_approve_and_generate_template,
    TemplateCreate,
    TemplateRequest,
    TemplateResponse,
    TemplateSuggestion,
    TaskConfig,
)

# Test client
client = TestClient(app)

# Mock data
TEST_ADMIN_TOKEN = "admin_token"
TEST_ANALYST_TOKEN = "analyst_token"
TEST_USER_ID = "test_user_id"

SAMPLE_TEMPLATE_DATA = {
    "name": "test_template",
    "description": "A test template",
    "agents": ["agent1", "agent2"],
    "tasks": [
        {"description": "task1", "expected_output": "output1", "agent_id": "agent1"}
    ],
    "process_type": "sequential",
    "verbose": True,
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
    "created_by": TEST_USER_ID,
    "sla_seconds": 600,
    "hitl_triggers": ["high_risk"],
}

SAMPLE_TEMPLATE_YAML = yaml.dump(SAMPLE_TEMPLATE_DATA, default_flow_style=False)


# Fixtures for mocking dependencies
@pytest.fixture
def mock_require_roles():
    """Mock the require_roles decorator to control access."""
    with patch("backend.api.v1.templates.require_roles") as mock_decorator:
        mock_decorator.return_value = lambda func: func  # Allow all by default
        yield mock_decorator


@pytest.fixture
def mock_admin_auth(mock_require_roles):
    """Mock authentication for an admin user."""
    mock_require_roles.return_value = lambda func: func  # Allow all
    with patch("backend.api.v1.templates.get_current_user") as mock_get_user:
        mock_user = MagicMock()
        mock_user.id = TEST_USER_ID
        mock_user.role = Roles.ADMIN
        mock_get_user.return_value = mock_user
        yield


@pytest.fixture
def mock_analyst_auth(mock_require_roles):
    """Mock authentication for an analyst user."""
    mock_require_roles.return_value = lambda func: func  # Allow all
    with patch("backend.api.v1.templates.get_current_user") as mock_get_user:
        mock_user = MagicMock()
        mock_user.id = TEST_USER_ID
        mock_user.role = Roles.ANALYST
        mock_get_user.return_value = mock_user
        yield


@pytest.fixture
def mock_no_auth(mock_require_roles):
    """Mock authentication failure."""
    mock_require_roles.side_effect = Exception("Unauthorized")
    yield


@pytest.fixture
def mock_crew_factory():
    """Mock CrewFactory."""
    with patch("backend.api.v1.templates.CrewFactory") as mock_factory_class:
        mock_factory = MagicMock()
        mock_factory_class.return_value = mock_factory
        yield mock_factory


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with patch("backend.api.v1.templates.Path") as mock_path_class, \
         patch("backend.api.v1.templates.open", new_callable=mock_open) as mock_file, \
         patch("os.makedirs") as mock_makedirs, \
         patch("os.path.exists") as mock_exists:
        
        # Setup mock path behavior
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.parent = mock_path_instance
        mock_path_instance.stem = "test_template"
        mock_path_instance.glob.return_value = [
            MagicMock(stem="template1"),
            MagicMock(stem="template2"),
        ]
        
        # Setup mock exists behavior
        mock_exists.return_value = True
        
        yield {
            "path": mock_path_class,
            "path_instance": mock_path_instance,
            "file": mock_file,
            "makedirs": mock_makedirs,
            "exists": mock_exists,
        }


# Tests for helper functions
class TestHelperFunctions:
    """Tests for template helper functions."""
    
    def test_get_template_path(self):
        """Test get_template_path function."""
        template_id = "test_template"
        path = get_template_path(template_id)
        assert isinstance(path, Path)
        assert path.name == f"{template_id}.yaml"
        assert str(AGENT_CONFIGS_CREWS_DIR) in str(path)
    
    def test_template_exists(self, mock_file_system):
        """Test template_exists function."""
        # Test template exists
        mock_file_system["path_instance"].exists.return_value = True
        assert template_exists("test_template") is True
        
        # Test template doesn't exist
        mock_file_system["path_instance"].exists.return_value = False
        assert template_exists("nonexistent_template") is False
    
    def test_load_template(self, mock_file_system):
        """Test load_template function."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        
        # Test successful load
        mock_file_system["path_instance"].exists.return_value = True
        template_data = load_template("test_template")
        assert template_data == SAMPLE_TEMPLATE_DATA
        
        # Test template not found
        mock_file_system["path_instance"].exists.return_value = False
        with pytest.raises(Exception, match="not found"):
            load_template("nonexistent_template")
        
        # Test load error
        mock_file_system["path_instance"].exists.return_value = True
        mock_file_system["file"].side_effect = Exception("File error")
        with pytest.raises(Exception, match="Failed to load template"):
            load_template("error_template")
    
    def test_save_template(self, mock_file_system):
        """Test save_template function."""
        # Test successful save
        save_template("test_template", SAMPLE_TEMPLATE_DATA)
        mock_file_system["makedirs"].assert_called_once()
        mock_file_system["file"].assert_called_once()
        
        # Test save error
        mock_file_system["file"].side_effect = Exception("File error")
        with pytest.raises(Exception, match="Failed to save template"):
            save_template("error_template", SAMPLE_TEMPLATE_DATA)
    
    def test_delete_template_file(self, mock_file_system):
        """Test delete_template_file function."""
        # Test successful delete
        mock_file_system["path_instance"].exists.return_value = True
        delete_template_file("test_template")
        mock_file_system["path_instance"].unlink.assert_called_once()
        
        # Test template not found
        mock_file_system["path_instance"].exists.return_value = False
        with pytest.raises(Exception, match="not found"):
            delete_template_file("nonexistent_template")
        
        # Test delete error
        mock_file_system["path_instance"].exists.return_value = True
        mock_file_system["path_instance"].unlink.side_effect = Exception("Delete error")
        with pytest.raises(Exception, match="Failed to delete template"):
            delete_template_file("error_template")
    
    def test_template_to_response(self):
        """Test template_to_response function."""
        response = template_to_response("test_template", SAMPLE_TEMPLATE_DATA)
        assert isinstance(response, TemplateResponse)
        assert response.id == "test_template"
        assert response.name == SAMPLE_TEMPLATE_DATA["name"]
        assert response.description == SAMPLE_TEMPLATE_DATA["description"]
        assert response.agents == SAMPLE_TEMPLATE_DATA["agents"]
        assert len(response.tasks) == 1
        assert response.tasks[0].description == "task1"
        assert response.process_type == SAMPLE_TEMPLATE_DATA["process_type"]
        assert response.verbose == SAMPLE_TEMPLATE_DATA["verbose"]
        assert response.created_at == SAMPLE_TEMPLATE_DATA["created_at"]
        assert response.updated_at == SAMPLE_TEMPLATE_DATA["updated_at"]
        assert response.created_by == SAMPLE_TEMPLATE_DATA["created_by"]
        assert response.sla_seconds == SAMPLE_TEMPLATE_DATA["sla_seconds"]
        assert response.hitl_triggers == SAMPLE_TEMPLATE_DATA["hitl_triggers"]


# Tests for template suggestions
class TestTemplateSuggestions:
    """Tests for template suggestions endpoints and functions."""
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_fraud(self):
        """Test generate_template_suggestions with fraud use case."""
        suggestions = await generate_template_suggestions("Investigate potential fraud in transactions")
        assert len(suggestions) == 2
        assert suggestions[0].name == "fraud_investigation"
        assert "fraud" in suggestions[0].description.lower()
        assert "fraud_pattern_hunter" in suggestions[0].agents
        assert "PatternLibraryTool" in suggestions[0].tools
        assert suggestions[0].confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_crypto(self):
        """Test generate_template_suggestions with crypto use case."""
        suggestions = await generate_template_suggestions("Analyze bitcoin transactions for suspicious patterns")
        assert len(suggestions) == 2
        assert suggestions[0].name == "crypto_investigation"
        assert "crypto" in suggestions[0].description.lower()
        assert "blockchain_detective" in suggestions[0].agents
        assert "CryptoAnomalyTool" in suggestions[0].tools
        assert suggestions[0].confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_analysis(self):
        """Test generate_template_suggestions with analysis use case."""
        suggestions = await generate_template_suggestions("Analyze transaction patterns")
        assert len(suggestions) == 2
        assert suggestions[0].name == "detailed_analysis"
        assert "analysis" in suggestions[0].description.lower()
        assert "code_analyst" in suggestions[0].agents
        assert "CodeGenTool" in suggestions[0].tools
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_compliance(self):
        """Test generate_template_suggestions with compliance use case."""
        suggestions = await generate_template_suggestions("Review transactions for compliance with AML regulations")
        assert len(suggestions) == 2
        assert suggestions[0].name == "compliance_review"
        assert "compliance" in suggestions[0].description.lower()
        assert "compliance_checker" in suggestions[0].agents
        assert "PolicyDocsTool" in suggestions[0].tools
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_default(self):
        """Test generate_template_suggestions with generic use case."""
        suggestions = await generate_template_suggestions("General investigation")
        assert len(suggestions) == 1
        assert suggestions[0].name == "default_investigation"
        assert "investigation" in suggestions[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_generate_template_suggestions_with_name(self):
        """Test generate_template_suggestions with custom template name."""
        suggestions = await generate_template_suggestions("Investigate potential fraud", "custom_template")
        assert len(suggestions) == 2
        assert suggestions[0].name == "custom_template"
    
    def test_get_template_suggestions_endpoint(self, mock_analyst_auth):
        """Test GET /api/v1/templates/suggestions endpoint."""
        response = client.get("/api/v1/templates/suggestions?use_case=Investigate+potential+fraud")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0
        assert "name" in data["suggestions"][0]
        assert "description" in data["suggestions"][0]
        assert "agents" in data["suggestions"][0]
        assert "tools" in data["suggestions"][0]
        assert "confidence" in data["suggestions"][0]
    
    def test_get_template_suggestions_with_name(self, mock_analyst_auth):
        """Test GET /api/v1/templates/suggestions with template_name."""
        response = client.get("/api/v1/templates/suggestions?use_case=Investigate+potential+fraud&template_name=custom_template")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["suggestions"][0]["name"] == "custom_template"
    
    def test_get_template_suggestions_unauthorized(self, mock_no_auth):
        """Test GET /api/v1/templates/suggestions with unauthorized user."""
        with pytest.raises(Exception, match="Unauthorized"):
            client.get("/api/v1/templates/suggestions?use_case=test")
    
    def test_get_template_suggestions_error(self, mock_analyst_auth):
        """Test GET /api/v1/templates/suggestions with error."""
        with patch("backend.api.v1.templates.generate_template_suggestions", side_effect=Exception("Test error")):
            response = client.get("/api/v1/templates/suggestions?use_case=test")
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "error" in response.json()


# Tests for template generation
class TestTemplateGeneration:
    """Tests for template generation functions."""
    
    @pytest.mark.asyncio
    async def test_generate_template_from_use_case(self):
        """Test generate_template_from_use_case function."""
        template = await generate_template_from_use_case("Investigate potential fraud", "fraud_template")
        assert template["name"] == "fraud_template"
        assert "fraud" in template["description"].lower()
        assert len(template["agents"]) > 0
        assert len(template["tasks"]) > 0
        assert template["process_type"] == "sequential"
        assert template["verbose"] is True
        assert "created_at" in template
        assert "updated_at" in template
        assert "sla_seconds" in template
        assert "hitl_triggers" in template
    
    @pytest.mark.asyncio
    async def test_auto_approve_and_generate_template(self, mock_file_system, mock_crew_factory):
        """Test auto_approve_and_generate_template function."""
        with patch("backend.api.v1.templates.generate_template_from_use_case", return_value=SAMPLE_TEMPLATE_DATA):
            await auto_approve_and_generate_template("request_id", "test_template", "test use case", TEST_USER_ID)
            mock_file_system["makedirs"].assert_called_once()
            mock_file_system["file"].assert_called_once()
            mock_crew_factory.reload.assert_called_once()


# Tests for CRUD operations
class TestCreateTemplate:
    """Tests for template creation endpoint."""
    
    def test_create_template_success(self, mock_admin_auth, mock_file_system, mock_crew_factory):
        """Test POST /api/v1/templates with valid data."""
        template_data = {
            "name": "New Template",
            "description": "A new test template",
            "agents": ["agent1", "agent2"],
            "tasks": [
                {"description": "task1", "expected_output": "output1", "agent_id": "agent1"}
            ],
            "process_type": "sequential",
            "verbose": True,
            "sla_seconds": 600,
            "hitl_triggers": ["high_risk"]
        }
        
        # Mock template doesn't exist
        mock_file_system["path_instance"].exists.return_value = False
        
        response = client.post("/api/v1/templates", json=template_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        data = response.json()
        assert data["name"] == template_data["name"]
        assert data["description"] == template_data["description"]
        assert data["agents"] == template_data["agents"]
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["description"] == "task1"
        assert data["process_type"] == template_data["process_type"]
        assert data["verbose"] == template_data["verbose"]
        assert data["sla_seconds"] == template_data["sla_seconds"]
        assert data["hitl_triggers"] == template_data["hitl_triggers"]
        
        # Verify file operations and CrewFactory reload
        mock_file_system["makedirs"].assert_called_once()
        mock_file_system["file"].assert_called_once()
        mock_crew_factory.reload.assert_called_once()
    
    def test_create_template_invalid_name(self, mock_admin_auth):
        """Test POST /api/v1/templates with invalid name."""
        template_data = {
            "name": "Invalid@Name",  # Contains special character
            "description": "A test template with invalid name",
            "agents": ["agent1"],
            "process_type": "sequential",
            "verbose": True
        }
        
        response = client.post("/api/v1/templates", json=template_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "name" in response.json()["detail"][0]["loc"]
    
    def test_create_template_invalid_process_type(self, mock_admin_auth):
        """Test POST /api/v1/templates with invalid process_type."""
        template_data = {
            "name": "Valid_Name",
            "description": "A test template with invalid process type",
            "agents": ["agent1"],
            "process_type": "invalid_type",  # Invalid process type
            "verbose": True
        }
        
        response = client.post("/api/v1/templates", json=template_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "process_type" in response.json()["detail"][0]["loc"]
    
    def test_create_template_conflict(self, mock_admin_auth, mock_file_system):
        """Test POST /api/v1/templates with existing template."""
        template_data = {
            "name": "Existing_Template",
            "description": "A test template that already exists",
            "agents": ["agent1"],
            "process_type": "sequential",
            "verbose": True
        }
        
        # Mock template exists
        mock_file_system["path_instance"].exists.return_value = True
        
        response = client.post("/api/v1/templates", json=template_data)
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"]
    
    def test_create_template_unauthorized(self, mock_analyst_auth):
        """Test POST /api/v1/templates with analyst role (not admin)."""
        # Override mock_require_roles to check for ADMIN
        with patch("backend.api.v1.templates.require_roles") as mock_decorator:
            # Make require_roles actually check for ADMIN role
            def check_roles(allowed_roles):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if Roles.ADMIN not in kwargs.get("roles", []):
                            raise Exception("Unauthorized - Admin required")
                        return func(*args, **kwargs)
                    return wrapper
                return decorator
            
            mock_decorator.side_effect = check_roles
            
            template_data = {
                "name": "Analyst_Template",
                "description": "A test template created by analyst",
                "agents": ["agent1"],
                "process_type": "sequential",
                "verbose": True
            }
            
            with pytest.raises(Exception, match="Unauthorized - Admin required"):
                client.post("/api/v1/templates", json=template_data)
    
    def test_create_template_error(self, mock_admin_auth, mock_file_system):
        """Test POST /api/v1/templates with error during save."""
        template_data = {
            "name": "Error_Template",
            "description": "A test template that causes an error",
            "agents": ["agent1"],
            "process_type": "sequential",
            "verbose": True
        }
        
        # Mock template doesn't exist but save fails
        mock_file_system["path_instance"].exists.return_value = False
        mock_file_system["file"].side_effect = Exception("Save error")
        
        response = client.post("/api/v1/templates", json=template_data)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to save template" in response.json()["detail"]


class TestGetTemplate:
    """Tests for template retrieval endpoint."""
    
    def test_get_template_success(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates/{template_id} with existing template."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        mock_file_system["path_instance"].exists.return_value = True
        
        response = client.get("/api/v1/templates/test_template")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == "test_template"
        assert data["name"] == SAMPLE_TEMPLATE_DATA["name"]
        assert data["description"] == SAMPLE_TEMPLATE_DATA["description"]
        assert data["agents"] == SAMPLE_TEMPLATE_DATA["agents"]
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["description"] == "task1"
    
    def test_get_template_not_found(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates/{template_id} with non-existent template."""
        mock_file_system["path_instance"].exists.return_value = False
        
        response = client.get("/api/v1/templates/nonexistent_template")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]
    
    def test_get_template_error(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates/{template_id} with error during load."""
        mock_file_system["path_instance"].exists.return_value = True
        mock_file_system["file"].side_effect = Exception("Load error")
        
        response = client.get("/api/v1/templates/error_template")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to load template" in response.json()["detail"]
    
    def test_get_template_unauthorized(self, mock_no_auth):
        """Test GET /api/v1/templates/{template_id} with unauthorized user."""
        with pytest.raises(Exception, match="Unauthorized"):
            client.get("/api/v1/templates/test_template")


class TestUpdateTemplate:
    """Tests for template update endpoint."""
    
    def test_update_template_success(self, mock_admin_auth, mock_file_system, mock_crew_factory):
        """Test PUT /api/v1/templates/{template_id} with valid data."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        mock_file_system["path_instance"].exists.return_value = True
        
        update_data = {
            "description": "Updated description",
            "agents": ["agent1", "agent2", "agent3"],
            "tasks": [
                {"description": "new_task", "expected_output": "new_output", "agent_id": "agent3"}
            ],
            "process_type": "hierarchical",
            "verbose": False,
            "sla_seconds": 1200,
            "hitl_triggers": ["high_risk", "medium_risk"]
        }
        
        response = client.put("/api/v1/templates/test_template", json=update_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == "test_template"
        assert data["description"] == update_data["description"]
        assert data["agents"] == update_data["agents"]
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["description"] == "new_task"
        assert data["process_type"] == update_data["process_type"]
        assert data["verbose"] == update_data["verbose"]
        assert data["sla_seconds"] == update_data["sla_seconds"]
        assert data["hitl_triggers"] == update_data["hitl_triggers"]
        
        # Verify file operations and CrewFactory reload
        mock_file_system["file"].assert_called()
        mock_crew_factory.reload.assert_called_once()
    
    def test_update_template_partial(self, mock_admin_auth, mock_file_system, mock_crew_factory):
        """Test PUT /api/v1/templates/{template_id} with partial update."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        mock_file_system["path_instance"].exists.return_value = True
        
        # Only update description and verbose
        update_data = {
            "description": "Partially updated description",
            "verbose": False
        }
        
        response = client.put("/api/v1/templates/test_template", json=update_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == "test_template"
        assert data["description"] == update_data["description"]
        assert data["verbose"] == update_data["verbose"]
        # Other fields should remain unchanged
        assert data["agents"] == SAMPLE_TEMPLATE_DATA["agents"]
        assert data["process_type"] == SAMPLE_TEMPLATE_DATA["process_type"]
        
        # Verify file operations and CrewFactory reload
        mock_file_system["file"].assert_called()
        mock_crew_factory.reload.assert_called_once()
    
    def test_update_template_not_found(self, mock_admin_auth, mock_file_system):
        """Test PUT /api/v1/templates/{template_id} with non-existent template."""
        mock_file_system["path_instance"].exists.return_value = False
        
        update_data = {"description": "Updated description"}
        
        response = client.put("/api/v1/templates/nonexistent_template", json=update_data)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]
    
    def test_update_template_invalid_process_type(self, mock_admin_auth):
        """Test PUT /api/v1/templates/{template_id} with invalid process_type."""
        update_data = {
            "process_type": "invalid_type"  # Invalid process type
        }
        
        response = client.put("/api/v1/templates/test_template", json=update_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "process_type" in response.json()["detail"][0]["loc"]
    
    def test_update_template_error(self, mock_admin_auth, mock_file_system):
        """Test PUT /api/v1/templates/{template_id} with error during save."""
        # Setup mock file content for initial load
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        mock_file_system["path_instance"].exists.return_value = True
        
        # Make save operation fail
        with patch("backend.api.v1.templates.save_template", side_effect=Exception("Save error")):
            update_data = {"description": "Updated description"}
            
            response = client.put("/api/v1/templates/test_template", json=update_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to update template" in response.json()["detail"]
    
    def test_update_template_unauthorized(self, mock_analyst_auth):
        """Test PUT /api/v1/templates/{template_id} with analyst role (not admin)."""
        # Override mock_require_roles to check for ADMIN
        with patch("backend.api.v1.templates.require_roles") as mock_decorator:
            # Make require_roles actually check for ADMIN role
            def check_roles(allowed_roles):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if Roles.ADMIN not in kwargs.get("roles", []):
                            raise Exception("Unauthorized - Admin required")
                        return func(*args, **kwargs)
                    return wrapper
                return decorator
            
            mock_decorator.side_effect = check_roles
            
            update_data = {"description": "Updated by analyst"}
            
            with pytest.raises(Exception, match="Unauthorized - Admin required"):
                client.put("/api/v1/templates/test_template", json=update_data)


class TestDeleteTemplate:
    """Tests for template deletion endpoint."""
    
    def test_delete_template_success(self, mock_admin_auth, mock_file_system, mock_crew_factory):
        """Test DELETE /api/v1/templates/{template_id} with existing template."""
        mock_file_system["path_instance"].exists.return_value = True
        
        response = client.delete("/api/v1/templates/test_template")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify file operations and CrewFactory reload
        mock_file_system["path_instance"].unlink.assert_called_once()
        mock_crew_factory.reload.assert_called_once()
    
    def test_delete_template_not_found(self, mock_admin_auth, mock_file_system):
        """Test DELETE /api/v1/templates/{template_id} with non-existent template."""
        mock_file_system["path_instance"].exists.return_value = False
        
        response = client.delete("/api/v1/templates/nonexistent_template")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]
    
    def test_delete_template_error(self, mock_admin_auth, mock_file_system):
        """Test DELETE /api/v1/templates/{template_id} with error during delete."""
        mock_file_system["path_instance"].exists.return_value = True
        mock_file_system["path_instance"].unlink.side_effect = Exception("Delete error")
        
        response = client.delete("/api/v1/templates/error_template")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to delete template" in response.json()["detail"]
    
    def test_delete_template_unauthorized(self, mock_analyst_auth):
        """Test DELETE /api/v1/templates/{template_id} with analyst role (not admin)."""
        # Override mock_require_roles to check for ADMIN
        with patch("backend.api.v1.templates.require_roles") as mock_decorator:
            # Make require_roles actually check for ADMIN role
            def check_roles(allowed_roles):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if Roles.ADMIN not in kwargs.get("roles", []):
                            raise Exception("Unauthorized - Admin required")
                        return func(*args, **kwargs)
                    return wrapper
                return decorator
            
            mock_decorator.side_effect = check_roles
            
            with pytest.raises(Exception, match="Unauthorized - Admin required"):
                client.delete("/api/v1/templates/test_template")


class TestListTemplates:
    """Tests for template listing endpoint."""
    
    def test_list_templates_success(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates with default pagination."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        
        # Mock Path.glob to return 2 template files
        mock_path_instance = mock_file_system["path_instance"]
        mock_path_instance.glob.return_value = [
            MagicMock(stem="template1"),
            MagicMock(stem="template2")
        ]
        
        response = client.get("/api/v1/templates")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "templates" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert data["total"] == 2
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert len(data["templates"]) == 2
    
    def test_list_templates_custom_pagination(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates with custom pagination."""
        # Setup mock file content
        mock_file_system["file"].return_value.__enter__.return_value.read.return_value = SAMPLE_TEMPLATE_YAML
        
        # Mock Path.glob to return 5 template files
        mock_path_instance = mock_file_system["path_instance"]
        mock_path_instance.glob.return_value = [
            MagicMock(stem=f"template{i}") for i in range(1, 6)
        ]
        
        response = client.get("/api/v1/templates?page=2&page_size=2")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total"] == 5
        assert data["page"] == 2
        assert data["page_size"] == 2
        assert len(data["templates"]) == 2  # Page 2 with size 2 should have 2 items
    
    def test_list_templates_empty(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates with no templates."""
        # Mock Path.glob to return empty list
        mock_path_instance = mock_file_system["path_instance"]
        mock_path_instance.glob.return_value = []
        
        response = client.get("/api/v1/templates")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total"] == 0
        assert len(data["templates"]) == 0
    
    def test_list_templates_error(self, mock_analyst_auth, mock_file_system):
        """Test GET /api/v1/templates with error during listing."""
        # Make glob operation fail
        mock_path_instance = mock_file_system["path_instance"]
        mock_path_instance.glob.side_effect = Exception("Glob error")
        
        response = client.get("/api/v1/templates")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to list templates" in response.json()["detail"]
    
    def test_list_templates_unauthorized(self, mock_no_auth):
        """Test GET /api/v1/templates with unauthorized user."""
        with pytest.raises(Exception, match="Unauthorized"):
            client.get("/api/v1/templates")


class TestTemplateRequest:
    """Tests for template request endpoint."""
    
    def test_create_template_request_analyst(self, mock_analyst_auth):
        """Test POST /api/v1/templates/request by analyst."""
        request_data = {
            "name": "Requested_Template",
            "use_case": "Investigate suspicious transactions",
            "sla_requirement": "Must complete within 10 minutes",
            "priority": "high"
        }
        
        response = client.post("/api/v1/templates/request", json=request_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        data = response.json()
        assert data["name"] == request_data["name"]
        assert data["use_case"] == request_data["use_case"]
        assert data["sla_requirement"] == request_data["sla_requirement"]
        assert data["priority"] == request_data["priority"]
        assert data["status"] == "pending"
        assert "created_at" in data
        assert "updated_at" in data
        assert "id" in data
    
    def test_create_template_request_admin(self, mock_admin_auth, mock_file_system, mock_crew_factory):
        """Test POST /api/v1/templates/request by admin with auto-approval."""
        request_data = {
            "name": "Admin_Template",
            "use_case": "Investigate suspicious transactions",
            "sla_requirement": "Must complete within 10 minutes",
            "priority": "high"
        }
        
        # Mock auto_approve_and_generate_template
        with patch("backend.api.v1.templates.auto_approve_and_generate_template") as mock_auto_approve:
            response = client.post("/api/v1/templates/request", json=request_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            data = response.json()
            assert data["name"] == request_data["name"]
            assert data["status"] == "pending"
            
            # Verify auto-approval task was added
            mock_auto_approve.assert_called_once()
    
    def test_create_template_request_default_priority(self, mock_analyst_auth):
        """Test POST /api/v1/templates/request with default priority."""
        request_data = {
            "name": "Default_Priority_Template",
            "use_case": "Investigate suspicious transactions",
            "sla_requirement": "Must complete within 10 minutes"
            # No priority specified
        }
        
        response = client.post("/api/v1/templates/request", json=request_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        data = response.json()
        assert data["priority"] == "medium"  # Default priority
    
    def test_create_template_request_invalid_name(self, mock_analyst_auth):
        """Test POST /api/v1/templates/request with invalid name."""
        request_data = {
            "name": "Invalid@Name",  # Contains special character
            "use_case": "Investigate suspicious transactions",
            "priority": "high"
        }
        
        response = client.post("/api/v1/templates/request", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "name" in response.json()["detail"][0]["loc"]
    
    def test_create_template_request_invalid_priority(self, mock_analyst_auth):
        """Test POST /api/v1/templates/request with invalid priority."""
        request_data = {
            "name": "Valid_Name",
            "use_case": "Investigate suspicious transactions",
            "priority": "invalid_priority"  # Invalid priority
        }
        
        response = client.post("/api/v1/templates/request", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "priority" in response.json()["detail"][0]["loc"]
    
    def test_create_template_request_error(self, mock_analyst_auth):
        """Test POST /api/v1/templates/request with error."""
        request_data = {
            "name": "Error_Template",
            "use_case": "Investigate suspicious transactions",
            "priority": "high"
        }
        
        with patch("uuid.uuid4", side_effect=Exception("UUID error")):
            response = client.post("/api/v1/templates/request", json=request_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to create template request" in response.json()["detail"]
    
    def test_create_template_request_unauthorized(self, mock_no_auth):
        """Test POST /api/v1/templates/request with unauthorized user."""
        request_data = {
            "name": "Unauthorized_Template",
            "use_case": "Investigate suspicious transactions",
            "priority": "high"
        }
        
        with pytest.raises(Exception, match="Unauthorized"):
            client.post("/api/v1/templates/request", json=request_data)
