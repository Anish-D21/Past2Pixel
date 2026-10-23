"""
Session History and Detail Endpoints for Past2Pixel API.
Enforces strict user-level access isolation and lifecycle management for processing sessions.
"""
from typing import List, Optional
import json
from fastapi import APIRouter, HTTPException, status, Depends, Response, Query

from ...auth.dependencies import get_current_user, get_optional_user
from ...db.mongodb import db_manager
from ...reporting.report_generator import session_report_generator
from ..schemas import SessionSummary, SessionDetailResponse

router = APIRouter(prefix="/sessions", tags=["Session History & Management"])


@router.get("", response_model=List[SessionSummary])
def list_my_sessions(current_user: dict = Depends(get_current_user)):
    """
    Returns history of restoration sessions owned by the authenticated user.
    Ordered chronologically (newest first).
    """
    user_id = str(current_user["_id"])
    docs = db_manager.list_user_sessions(user_id=user_id, limit=50)

    summaries = []
    for doc in docs:
        orig_meta = doc.get("original_metadata", {})
        res_res = doc.get("restoration_result") or {}
        damage_info = res_res.get("damage_detection") or {}

        summaries.append(
            SessionSummary(
                session_id=doc["session_id"],
                created_at=doc.get("created_at", ""),
                status=doc.get("status", "UNKNOWN"),
                filename=orig_meta.get("original_filename"),
                has_ocr=doc.get("ocr_result") is not None,
                has_restoration=doc.get("restoration_result") is not None,
                damaged_percentage=damage_info.get("damaged_pixel_percentage"),
            )
        )

    return summaries


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session_details(
    session_id: str,
    current_user: dict = Depends(get_optional_user),
):
    """
    Retrieves full image-specific processing session report, including
    OCR text, damage metrics, intermediate artifacts, and device timings.
    Enforces user isolation: users cannot access sessions owned by others.
    """
    session = db_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    # Ownership access control
    owner_id = session.get("user_id")
    if owner_id and owner_id != "anonymous":
        if not current_user or str(current_user["_id"]) != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You do not have permission to view this session.",
            )

    return SessionDetailResponse(
        session_id=session["session_id"],
        user_id=session.get("user_id"),
        created_at=session.get("created_at", ""),
        updated_at=session.get("updated_at", ""),
        status=session.get("status", "UNKNOWN"),
        original_metadata=session.get("original_metadata", {}),
        artifacts=session.get("artifacts", {}),
        ocr_result=session.get("ocr_result"),
        restoration_result=session.get("restoration_result"),
        text_predictions=session.get("text_predictions"),
        persistence_info=session.get("persistence_info", {}),
    )


@router.get("/{session_id}/report")
def download_session_report(
    session_id: str,
    format: str = Query("html", description="Report format: pdf, html, json, or markdown"),
    current_user: dict = Depends(get_optional_user),
):
    """
    Generates and returns an Explainable AI (XAI) laboratory report for the specified session.
    Supported formats: 'pdf', 'html', 'json', 'markdown' (or 'md').
    Enforces user ownership isolation: users cannot access reports of others' private sessions.
    """
    clean_session_id = session_id.strip()
    session = db_manager.get_session(clean_session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{clean_session_id}' not found.",
        )

    # Ownership check
    owner_id = session.get("user_id")
    if owner_id and owner_id != "anonymous":
        if not current_user or str(current_user["_id"]) != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You do not have permission to view the report for this session.",
            )

    fmt = format.lower().strip()
    if fmt == "pdf":
        pdf_bytes = session_report_generator.generate_pdf_report(session)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="past2pixel_xai_report_{clean_session_id}.pdf"'},
        )
    elif fmt == "json":
        json_data = session_report_generator.generate_json_report(session)
        return Response(
            content=json.dumps(json_data, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="past2pixel_xai_report_{clean_session_id}.json"'},
        )
    elif fmt in ["markdown", "md"]:
        md_text = session_report_generator.generate_markdown_report(session)
        return Response(
            content=md_text,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="past2pixel_xai_report_{clean_session_id}.md"'},
        )
    else:  # default to html
        html_text = session_report_generator.generate_html_report(session)
        return Response(
            content=html_text,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f'inline; filename="past2pixel_xai_report_{clean_session_id}.html"'},
        )


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Permanently deletes a restoration session and its disk artifacts.
    Enforces strict ownership: only the owner may delete a session.
    """
    user_id = str(current_user["_id"])
    session = db_manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    if session.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: You cannot delete another user's session.",
        )

    deleted = db_manager.delete_session(session_id=session_id, user_id=user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session from database.",
        )

    return {
        "status": "DELETED",
        "session_id": session_id,
        "message": "Session and all associated visual artifacts deleted successfully.",
    }
