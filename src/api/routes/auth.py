"""
User Authentication & Registration Endpoints for Past2Pixel API.
"""
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends

from ...core.config import settings
from ...auth.security import hash_password, verify_password, create_access_token
from ...auth.dependencies import get_current_user
from ...db.mongodb import db_manager
from ..schemas import UserRegisterRequest, UserLoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication & Users"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: UserRegisterRequest):
    """
    Registers a new user account with secure password hashing.
    Rejects duplicate emails or usernames with clear error messages.
    """
    # 1. Check existing email or username
    if db_manager.get_user_by_email(req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An account with email '{req.email}' is already registered.",
        )

    # 2. Hash password securely
    hashed_pwd = hash_password(req.password)

    user_doc = {
        "email": req.email.strip().lower(),
        "username": req.username.strip(),
        "password_hash": hashed_pwd,
    }

    try:
        saved_user = db_manager.create_user(user_doc)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user due to a database error.",
        )

    # 3. Issue initial JWT bearer token
    user_id = str(saved_user["_id"])
    access_token = create_access_token(
        data={"sub": user_id, "email": saved_user["email"], "username": saved_user["username"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    user_resp = UserResponse(
        id=user_id,
        email=saved_user["email"],
        username=saved_user["username"],
        created_at=saved_user["created_at"],
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=user_resp,
    )


@router.post("/login", response_model=TokenResponse)
def login_user(req: UserLoginRequest):
    """
    Authenticates a user via email or username and password.
    Returns a signed JWT bearer token upon successful verification.
    """
    identifier = req.username_or_email.strip().lower()
    user = db_manager.get_user_by_email(identifier)

    if not user:
        # Search by username in local/mongo store
        all_users = db_manager._read_local_store().get("users", []) if db_manager.storage_mode == "local_file_persistence" else []
        for u in all_users:
            if u.get("username", "").lower() == identifier:
                user = u
                break
        if not user and db_manager.storage_mode == "mongodb_atlas" and db_manager.db is not None:
            user = db_manager.db.users.find_one({"username": req.username_or_email.strip()})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password hash
    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user["_id"])
    access_token = create_access_token(
        data={"sub": user_id, "email": user["email"], "username": user["username"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    user_resp = UserResponse(
        id=user_id,
        email=user["email"],
        username=user["username"],
        created_at=user.get("created_at", ""),
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=user_resp,
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Returns the authenticated user's profile details."""
    return UserResponse(
        id=str(current_user["_id"]),
        email=current_user["email"],
        username=current_user["username"],
        created_at=current_user.get("created_at", ""),
    )
