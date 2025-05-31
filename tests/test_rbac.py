"""
Tests for Role-Based Access Control (RBAC) functionality.

This module contains tests for the RBAC decorators and utilities,
verifying that they correctly enforce role-based access control
on FastAPI endpoints.
"""

import pytest
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.auth.rbac import require_roles, has_roles, Roles, RoleSets
from backend.auth.jwt_handler import create_access_token


# Test app setup
app = FastAPI()

# Test endpoints with RBAC
@app.get("/admin-only")
@require_roles([Roles.ADMIN])
async def admin_only_endpoint(request: Request):
    return {"message": "Admin access granted", "user": request.state.user}

@app.get("/analyst-or-admin")
@require_roles([Roles.ANALYST, Roles.ADMIN])
async def analyst_or_admin_endpoint(request: Request):
    return {"message": "Analyst or Admin access granted", "user": request.state.user}

@app.get("/all-staff")
@require_roles(RoleSets.ALL_STAFF)
async def all_staff_endpoint(request: Request):
    return {"message": "Staff access granted", "user": request.state.user}

# Test endpoint with dependency-style RBAC
@app.get("/compliance-team")
async def compliance_team_endpoint(
    request: Request,
    _: bool = Depends(lambda req: has_roles(req, RoleSets.COMPLIANCE_TEAM))
):
    return {"message": "Compliance team access granted", "user": request.state.user}


# Fixtures for different user roles
@pytest.fixture
def admin_token():
    """Create a token for an admin user."""
    return create_access_token({"sub": "admin@example.com", "role": "admin"})

@pytest.fixture
def analyst_token():
    """Create a token for an analyst user."""
    return create_access_token({"sub": "analyst@example.com", "role": "analyst"})

@pytest.fixture
def compliance_token():
    """Create a token for a compliance officer."""
    return create_access_token({"sub": "compliance@example.com", "role": "compliance"})

@pytest.fixture
def user_token():
    """Create a token for a regular user."""
    return create_access_token({"sub": "user@example.com", "role": "user"})

@pytest.fixture
def no_role_token():
    """Create a token for a user with no role."""
    return create_access_token({"sub": "norole@example.com"})


# Test client setup
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# Mock middleware to set user in request state
@app.middleware("http")
async def mock_auth_middleware(request: Request, call_next):
    """Middleware to mock authentication and set user in request state."""
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        # Mock token validation - in a real app this would verify the token
        with patch("backend.auth.jwt_handler.decode_token") as mock_decode:
            # Different test tokens return different user data
            if token == "admin_token":
                mock_decode.return_value = {"sub": "admin@example.com", "role": "admin"}
            elif token == "analyst_token":
                mock_decode.return_value = {"sub": "analyst@example.com", "role": "analyst"}
            elif token == "compliance_token":
                mock_decode.return_value = {"sub": "compliance@example.com", "role": "compliance"}
            elif token == "user_token":
                mock_decode.return_value = {"sub": "user@example.com", "role": "user"}
            elif token == "no_role_token":
                mock_decode.return_value = {"sub": "norole@example.com"}
            else:
                # For actual JWT tokens from fixtures
                from backend.auth.jwt_handler import decode_token
                try:
                    user_data = decode_token(token)
                    request.state.user = user_data
                except Exception:
                    # Invalid token
                    pass
                return await call_next(request)
            
            # Set user in request state
            request.state.user = mock_decode.return_value
    
    return await call_next(request)


# Tests for RBAC decorator
def test_admin_only_with_admin(client):
    """Test that admin can access admin-only endpoint."""
    response = client.get("/admin-only", headers={"Authorization": "Bearer admin_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Admin access granted"
    assert response.json()["user"]["role"] == "admin"


def test_admin_only_with_analyst(client):
    """Test that analyst cannot access admin-only endpoint."""
    response = client.get("/admin-only", headers={"Authorization": "Bearer analyst_token"})
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_admin_only_with_no_auth(client):
    """Test that unauthenticated user cannot access admin-only endpoint."""
    response = client.get("/admin-only")
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_analyst_or_admin_with_analyst(client):
    """Test that analyst can access analyst-or-admin endpoint."""
    response = client.get("/analyst-or-admin", headers={"Authorization": "Bearer analyst_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Analyst or Admin access granted"
    assert response.json()["user"]["role"] == "analyst"


def test_analyst_or_admin_with_admin(client):
    """Test that admin can access analyst-or-admin endpoint."""
    response = client.get("/analyst-or-admin", headers={"Authorization": "Bearer admin_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Analyst or Admin access granted"
    assert response.json()["user"]["role"] == "admin"


def test_analyst_or_admin_with_user(client):
    """Test that regular user cannot access analyst-or-admin endpoint."""
    response = client.get("/analyst-or-admin", headers={"Authorization": "Bearer user_token"})
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_all_staff_with_admin(client):
    """Test that admin can access all-staff endpoint."""
    response = client.get("/all-staff", headers={"Authorization": "Bearer admin_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Staff access granted"


def test_all_staff_with_analyst(client):
    """Test that analyst can access all-staff endpoint."""
    response = client.get("/all-staff", headers={"Authorization": "Bearer analyst_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Staff access granted"


def test_all_staff_with_compliance(client):
    """Test that compliance officer can access all-staff endpoint."""
    response = client.get("/all-staff", headers={"Authorization": "Bearer compliance_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Staff access granted"


def test_all_staff_with_user(client):
    """Test that regular user cannot access all-staff endpoint."""
    response = client.get("/all-staff", headers={"Authorization": "Bearer user_token"})
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


# Tests for dependency-style RBAC
def test_compliance_team_with_compliance(client):
    """Test that compliance officer can access compliance-team endpoint."""
    response = client.get("/compliance-team", headers={"Authorization": "Bearer compliance_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Compliance team access granted"


def test_compliance_team_with_admin(client):
    """Test that admin can access compliance-team endpoint."""
    response = client.get("/compliance-team", headers={"Authorization": "Bearer admin_token"})
    assert response.status_code == 200
    assert response.json()["message"] == "Compliance team access granted"


def test_compliance_team_with_analyst(client):
    """Test that analyst cannot access compliance-team endpoint."""
    response = client.get("/compliance-team", headers={"Authorization": "Bearer analyst_token"})
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


# Tests with actual JWT tokens from fixtures
def test_with_actual_admin_token(client, admin_token):
    """Test with an actual JWT token for admin."""
    response = client.get("/admin-only", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    assert response.json()["message"] == "Admin access granted"


def test_with_actual_analyst_token(client, analyst_token):
    """Test with an actual JWT token for analyst."""
    response = client.get("/analyst-or-admin", headers={"Authorization": f"Bearer {analyst_token}"})
    assert response.status_code == 200
    assert response.json()["message"] == "Analyst or Admin access granted"


def test_with_actual_no_role_token(client, no_role_token):
    """Test with an actual JWT token for user with no role."""
    response = client.get("/admin-only", headers={"Authorization": f"Bearer {no_role_token}"})
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


# Edge case tests
def test_with_invalid_token(client):
    """Test with an invalid token."""
    response = client.get("/admin-only", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_with_malformed_auth_header(client):
    """Test with a malformed Authorization header."""
    response = client.get("/admin-only", headers={"Authorization": "NotBearer token"})
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


# Integration tests with mocked request objects
def test_require_roles_decorator_directly():
    """Test the require_roles decorator directly with a mocked request."""
    # Create a mock request with admin user
    mock_request = MagicMock()
    mock_request.state.user = {"role": "admin"}
    
    # Create a test function with the decorator
    @require_roles([Roles.ADMIN])
    async def test_func(request):
        return {"success": True}
    
    # Test with admin role (should pass)
    import asyncio
    result = asyncio.run(test_func(mock_request))
    assert result == {"success": True}
    
    # Test with non-admin role (should raise HTTPException)
    mock_request.state.user = {"role": "user"}
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(test_func(mock_request))
    assert excinfo.value.status_code == 403
    
    # Test with no user (should raise HTTPException)
    delattr(mock_request.state, "user")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(test_func(mock_request))
    assert excinfo.value.status_code == 401


def test_has_roles_dependency_directly():
    """Test the has_roles dependency directly with a mocked request."""
    # Create a mock request with admin user
    mock_request = MagicMock()
    mock_request.state.user = {"role": "admin"}
    
    # Test with admin role (should return True)
    import asyncio
    result = asyncio.run(has_roles(mock_request, [Roles.ADMIN]))
    assert result is True
    
    # Test with non-admin role (should raise HTTPException)
    mock_request.state.user = {"role": "user"}
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(has_roles(mock_request, [Roles.ADMIN]))
    assert excinfo.value.status_code == 403
    
    # Test with no user (should raise HTTPException)
    delattr(mock_request.state, "user")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(has_roles(mock_request, [Roles.ADMIN]))
    assert excinfo.value.status_code == 401


# Test custom error messages
def test_custom_error_message():
    """Test that custom error messages are used."""
    # Create a test app with custom error message
    test_app = FastAPI()
    
    @test_app.get("/custom-error")
    @require_roles([Roles.ADMIN], error_message="Custom access denied message")
    async def custom_error_endpoint(request: Request):
        return {"message": "Access granted"}
    
    # Add the mock middleware
    test_app.middleware("http")(mock_auth_middleware)
    
    # Create a test client
    test_client = TestClient(test_app)
    
    # Test with non-admin role
    response = test_client.get("/custom-error", headers={"Authorization": "Bearer analyst_token"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Custom access denied message"
