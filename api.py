from database import detections, documents, users, sites
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json
import re
from typing import List, Optional
import os
import shutil
import traceback
import time
import cv2
from datetime import datetime
from bson import ObjectId
from detect import detect_objects
from associate import associate_ppe
from violation_checker import check_violations, compute_compliance_rate, evaluate_worker
from llm_provider import describe as describe_llm
from pydantic import BaseModel
from langgraph_pipeline import workflow
from document_processor import extract_text
from knowledge_base import update_knowledge_base
from pdf_report_generator import generate_pdf_report  # ✅ FIXED: Correct import
from auth import (
    create_access_token,
    hash_password,
    verify_password,
    get_current_user,
    require_admin,
    seed_default_users,
)

app = FastAPI(
    title="Safesight API",
    description="AI-Based Construction PPE Detection API",
    version="2.0.0"
)

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Create Required Folders
# ----------------------------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
DOCUMENT_FOLDER = "documents"
REPORT_FOLDER = "reports"
VIDEO_FOLDER = "videos"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Safety caps for video processing — every frame is still run through YOLO,
# but a very long video would otherwise never finish / blow up storage.
# If a video has more frames than this, we sample evenly across its full
# length so the whole video is still represented, not just the first chunk.
MAX_VIDEO_FRAMES = 300
# Of the frames processed, only this many are kept as annotated result
# images (violations are prioritized), to keep the response/storage sane.
MAX_VIDEO_RESULT_FRAMES = 24
# When a video is long enough to be sampled (not every frame processed),
# look at up to this many consecutive frames around each sample point and
# keep the sharpest one instead of blindly using the first — motion blur
# in video frames is a major cause of missed/incorrect PPE detections.
BLUR_SEARCH_WINDOW = 4

# ----------------------------------------------------
# Seed default login accounts (admin/admin123, user/user123)
# ----------------------------------------------------
@app.on_event("startup")
def on_startup():
    seed_default_users()
    # Seed one default site so the Upload Center always has something to
    # assign detections to, even before an admin has created any sites.
    if sites.count_documents({}) == 0:
        sites.insert_one({"name": "Construction Site A", "created_at": str(datetime.now())})

# ----------------------------------------------------
# Serve Annotated Images and Documents
# ----------------------------------------------------
app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_FOLDER),
    name="outputs"
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

app.mount(
    "/documents",
    StaticFiles(directory=DOCUMENT_FOLDER),
    name="documents"
)

app.mount(
    "/videos",
    StaticFiles(directory=VIDEO_FOLDER),
    name="videos"
)

# ----------------------------------------------------
# Chat Request Model
# ----------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    search_scope: str = "all"          # all | selected
    selected_manuals: List[str] = []


class LoginRequest(BaseModel):
    username: str
    password: str


class SiteRequest(BaseModel):
    name: str


# ----------------------------------------------------
# Auth: Login
# ----------------------------------------------------
def _find_user_for_login(raw_username: str):
    """
    Look up a login by username, tolerating the two things people actually
    type: stray whitespace and the wrong capitalisation.

    The lookup used to be an exact match on whatever string arrived, so an
    account created as "Anisha" simply could not log in as "anisha" — the
    credentials were valid and the response was still "Invalid username or
    password". Usernames are identities here, not passwords; case is not a
    security boundary. The exact match is still tried first so an existing
    exact username always wins, and only then the case-insensitive fallback.
    """
    username = (raw_username or "").strip()
    if not username:
        return None

    user = users.find_one({"username": username})
    if user is not None:
        return user

    return users.find_one(
        {"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}
    )


@app.post("/auth/login")
def login(request: LoginRequest):
    user = _find_user_for_login(request.username)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    token = create_access_token(username=user["username"], role=user["role"])

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"]
    }


# ----------------------------------------------------
# Auth: Current logged-in user
# ----------------------------------------------------
@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "success": True,
        "username": current_user["username"],
        "role": current_user["role"]
    }


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.post("/auth/change-password")
def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user)
):
    """Self-service password change, available to any logged-in account."""
    user = users.find_one({"username": current_user["username"]})

    if not user or not verify_password(request.current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    if len(request.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    users.update_one(
        {"username": current_user["username"]},
        {"$set": {"password_hash": hash_password(request.new_password)}}
    )

    return {"success": True, "message": "Password updated successfully."}


# ----------------------------------------------------
# User Management (Admin only)
# ----------------------------------------------------
VALID_ROLES = ["admin", "user"]


def _serialize_user(u):
    return {
        "username": u["username"],
        "role": u["role"],
        "created_at": u.get("created_at"),
    }


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    password: Optional[str] = None


@app.get("/users")
def list_users(current_user: dict = Depends(require_admin)):
    all_users = list(users.find({}, {"_id": 0, "password_hash": 0}))
    return {"success": True, "count": len(all_users), "users": all_users}


@app.post("/users")
def create_user(request: CreateUserRequest, current_user: dict = Depends(require_admin)):
    username = request.username.strip()

    if not username or not request.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    if request.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Role must be one of {VALID_ROLES}.")

    # Case-insensitive, because login now resolves usernames case-
    # insensitively too — allowing both "Anisha" and "anisha" to exist would
    # make which account you land in depend on exact capitalisation.
    if users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}):
        raise HTTPException(status_code=409, detail="A user with that username already exists.")

    new_user = {
        "username": username,
        "password_hash": hash_password(request.password),
        "role": request.role,
        "created_at": str(datetime.now()),
    }
    users.insert_one(new_user)

    return {"success": True, "user": _serialize_user(new_user)}


@app.put("/users/{username}")
def update_user(
    username: str,
    request: UpdateUserRequest,
    current_user: dict = Depends(require_admin)
):
    target = users.find_one({"username": username})
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    updates = {}

    if request.role is not None:
        if request.role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail=f"Role must be one of {VALID_ROLES}.")
        # Prevent demoting the last remaining admin
        if target["role"] == "admin" and request.role != "admin":
            remaining_admins = users.count_documents({"role": "admin", "username": {"$ne": username}})
            if remaining_admins == 0:
                raise HTTPException(status_code=400, detail="Cannot demote the last remaining admin.")
        updates["role"] = request.role

    if request.password is not None:
        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        updates["password_hash"] = hash_password(request.password)

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    users.update_one({"username": username}, {"$set": updates})
    updated = users.find_one({"username": username})

    return {"success": True, "user": _serialize_user(updated)}


@app.delete("/users/{username}")
def delete_user(username: str, current_user: dict = Depends(require_admin)):
    target = users.find_one({"username": username})
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account while logged in.")

    if target["role"] == "admin":
        remaining_admins = users.count_documents({"role": "admin", "username": {"$ne": username}})
        if remaining_admins == 0:
            raise HTTPException(status_code=400, detail="Cannot delete the last remaining admin.")

    users.delete_one({"username": username})

    return {"success": True, "message": f"User '{username}' deleted."}


# ----------------------------------------------------
# Health Check
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "project": "Safesight",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    # Reported from the live configuration rather than a hardcoded string —
    # this endpoint feeds the AI Assistant's footer, and it previously
    # claimed "llama3.1" no matter which model was actually answering.
    llm_info = describe_llm()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "knowledge_base": "ready",
        "model": llm_info["model"],
        "llm_provider": llm_info["provider"],
        "llm_is_local": llm_info["is_local"],
        "embedding_model": "MiniLM",
        "vector_db": "FAISS"
    }


# ----------------------------------------------------
# Get Available Manuals Endpoint
# ----------------------------------------------------
@app.get("/manuals")
def get_manuals(current_user: dict = Depends(get_current_user)):
    """
    Fetch all available manual filenames from the database.
    Used by the frontend to populate the manual selection checkbox list.
    """
    try:
        # Query only the filename field, exclude _id
        docs = list(
            documents.find(
                {},
                {
                    "_id": 0,
                    "filename": 1,
                    "document_type": 1  # Optional: include document type for filtering
                }
            )
        )
        
        # Extract just the filenames
        manuals = [
            d["filename"]
            for d in docs
            if d.get("document_type") in ["PDF Manual", "DOCX Manual", "Text Document"]
        ]
        
        print(f"📚 Retrieved {len(manuals)} manuals from database")
        
        return {
            "success": True,
            "count": len(manuals),
            "manuals": manuals
        }
        
    except Exception as e:
        print(f"❌ Error fetching manuals: {e}")
        return {
            "success": False,
            "error": str(e),
            "manuals": []
        }


# ----------------------------------------------------
# Sites: List / Create / Delete
# ----------------------------------------------------
def _serialize_site(doc):
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name"),
        "created_at": doc.get("created_at"),
    }


@app.get("/sites")
def list_sites(current_user: dict = Depends(get_current_user)):
    """
    List all construction sites, used to populate the site selector in the
    Upload Center and the site-management admin page.
    """
    docs = list(sites.find().sort("name", 1))
    return {
        "success": True,
        "sites": [_serialize_site(d) for d in docs],
    }


@app.post("/sites")
def create_site(request: SiteRequest, current_user: dict = Depends(require_admin)):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Site name cannot be empty.")

    existing = sites.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
    if existing:
        raise HTTPException(status_code=400, detail="A site with this name already exists.")

    doc = {"name": name, "created_at": str(datetime.now())}
    result = sites.insert_one(doc)
    doc["_id"] = result.inserted_id
    return {"success": True, "site": _serialize_site(doc)}


@app.delete("/sites/{site_id}")
def delete_site(site_id: str, current_user: dict = Depends(require_admin)):
    try:
        result = sites.delete_one({"_id": ObjectId(site_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid site id.")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Site not found.")

    return {"success": True, "message": "Site deleted."}


# ----------------------------------------------------
# Multiple Image Detection Endpoint
# ----------------------------------------------------
@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(...),
    site_name: str = Form("Construction Site A"),
    current_user: dict = Depends(require_admin)
):
    images = []
    summary = {
        "total_images": 0,
        "total_workers": 0,
        "safe_workers": 0,
        "unsafe_workers": 0,
        "helmet": 0,
        "vest": 0,
        "gloves": 0,
        "boots": 0,
        "goggles": 0,
        "compliance": 0
    }
    logs = []

    for file in files:
        # -----------------------------
        # Save Uploaded Image
        # -----------------------------
        upload_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -----------------------------
        # Run YOLO Detection
        # -----------------------------
        results = detect_objects(upload_path)

        # -----------------------------
        # Save Annotated Image
        # -----------------------------
        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.filename
        )
        results[0].save(filename=output_path)

        # -----------------------------
        # Associate PPE with Workers
        # -----------------------------
        workers = associate_ppe(results)

        # ✅ DEBUG: Print associated workers
        print("\n===== AFTER associate_ppe =====")
        for w in workers:
            print(w)
        print("==============================")

        # -----------------------------
        # Check Violations / Build Worker Status
        # -----------------------------
        report = check_violations(workers)

        processed_workers = report["workers"]

        summary["total_workers"] += len(processed_workers)
        summary["safe_workers"] += report["summary"]["safe_workers"]
        summary["unsafe_workers"] += report["summary"]["unsafe_workers"]

        summary["helmet"] += report["summary"]["helmet"]
        summary["vest"] += report["summary"]["vest"]
        summary["gloves"] += report["summary"]["gloves"]
        summary["boots"] += report["summary"]["boots"]
        summary["goggles"] += report["summary"]["goggles"]

        summary["total_images"] += 1

        # ✅ FIX: Calculate per-image metrics
        image_safe = report["summary"]["safe_workers"]
        image_unsafe = report["summary"]["unsafe_workers"]
        image_total = image_safe + image_unsafe

        # % of required PPE items present, not safe/total workers — see
        # compute_compliance_rate for why (single-worker images otherwise
        # always read as a binary 0% or 100%).
        image_compliance = compute_compliance_rate(processed_workers)

        # Extract unique missing PPE from all workers
        missing_ppe = []
        for worker in processed_workers:
            missing_ppe.extend(worker.get("missing", []))

        missing_ppe = list(set(missing_ppe))

        images.append(
            {
                "name": file.filename,
                "site_name": site_name,
                "original": f"http://127.0.0.1:8000/uploads/{file.filename}",
                "annotated": f"http://127.0.0.1:8000/outputs/{file.filename}",
                "workers": processed_workers,
                # ✅ NEW FIELDS
                "safe_workers": image_safe,
                "unsafe_workers": image_unsafe,
                "total_workers": image_total,
                "compliance_rate": image_compliance,
                "missing": missing_ppe,
                "status": "Safe" if image_unsafe == 0 else "Unsafe",
            }
        )

        # -----------------------------
        # Kept for lightweight logging only
        # (NOT used for the stored record anymore)
        # -----------------------------
        logs.append(
            {
                "timestamp": str(datetime.now()),
                "image": file.filename,
                "workers": len(processed_workers),
                "unsafe": len(
                    [
                        w
                        for w in processed_workers
                        if w["status"] == "Unsafe"
                    ]
                ),
            }
        )

    # ----------------------------------------------------
    # Calculate Overall Compliance
    # ----------------------------------------------------
    total_required_ppe = summary["total_workers"] * 5
    total_missing_ppe = (
        summary["helmet"]
        + summary["vest"]
        + summary["gloves"]
        + summary["boots"]
        + summary["goggles"]
    )

    if total_required_ppe > 0:
        summary["compliance"] = round(
            ((total_required_ppe - total_missing_ppe)
             / total_required_ppe) * 100,
            2
        )
    else:
        summary["compliance"] = 100.0

    # ----------------------------------------------------
    # Save Detection History to MongoDB
    # ----------------------------------------------------
    # IMPORTANT: store `images` (full worker objects), not `logs`
    # (which only holds a worker COUNT, not the worker list itself)
    # ----------------------------------------------------
    record = {
        "timestamp": str(datetime.now()),
        "summary": summary,
        "images": images,
        "source_type": "image",
        "acknowledged": False,
    }

    try:
        detections.insert_one(record)
        print("Detection history saved to MongoDB.")
    except Exception as e:
        print(f"MongoDB Error: {e}")

    # ----------------------------------------------------
    # Return Response
    # ----------------------------------------------------
    return {
        "summary": summary,
        "images": images
    }


# ----------------------------------------------------
# Video Detection Endpoint
# ----------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def _compliance_pct(summary):
    """Same "% of required PPE actually present" formula used everywhere else."""
    total_required = summary["total_workers"] * 5
    total_missing = (
        summary["helmet"] + summary["vest"] + summary["gloves"]
        + summary["boots"] + summary["goggles"]
    )
    if total_required <= 0:
        return 100.0
    return round(((total_required - total_missing) / total_required) * 100, 2)


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0


PPE_ITEMS = ("helmet", "vest", "gloves", "boots", "goggles")

# How many of a tracked person's most recent sightings to look at when
# deciding their current PPE status. See _aggregate_tracked_summary.
RECENT_OBSERVATION_WINDOW = 5


def _aggregate_tracked_summary(tracks, frames_analyzed, fps=None):
    """
    Turns per-frame person tracks into a "how many actual workers were in
    this video" summary, instead of naively summing every frame's detections
    (which would count the same person again in every frame they appear in).

    Verdicts come from the same cumulative per-worker rows the checklist
    table shows (_cumulative_tracked_workers) and the same Safe/Unsafe
    threshold everything else uses (violation_checker.evaluate_worker).
    That agreement is the whole point of routing it through there.

    It previously ran its own rule — majority vote over each person's last
    few sightings, and "any missing item at all = unsafe" — which disagreed
    with the table twice over. A worker could show three ticked items in
    the checklist and still be counted as a violation in the summary
    cards, because (a) the cards were only looking at the tail of the
    track, so PPE donned earlier and briefly out of view didn't count, and
    (b) the cards ignored the 3-of-5 threshold that decides Safe/Unsafe
    everywhere else, so even 4 of 5 items read as a violation. Two numbers
    on the same screen contradicting each other is worse than either rule
    being slightly off, so there is now only one rule.

    Consequence worth knowing: because the cumulative view never expires an
    item, PPE that is removed later still counts as present here. Per-frame
    status (each frame's own Safe/Unsafe badge) is still the place to see
    what someone is wearing at a given moment.
    """
    counts = {item: 0 for item in PPE_ITEMS}
    safe_workers = 0
    unsafe_workers = 0

    for worker in _cumulative_tracked_workers(tracks, fps):
        if worker["status"] == "Safe":
            safe_workers += 1
        else:
            unsafe_workers += 1

        # Missing items are tallied for every worker, safe or not — a Safe
        # worker missing goggles still needs to show up in the site's
        # missing-PPE breakdown. This matches check_violations.
        for item in PPE_ITEMS:
            if not worker[item]:
                counts[item] += 1

    summary = {
        "total_images": frames_analyzed,
        "total_workers": safe_workers + unsafe_workers,
        "safe_workers": safe_workers,
        "unsafe_workers": unsafe_workers,
        **counts,
        "compliance": 0,
    }
    summary["compliance"] = _compliance_pct(summary)
    return summary


def _smoothed_worker_ppe(track_id, tracks):
    """
    A tracked worker's per-item PPE flags (helmet/vest/gloves/boots/
    goggles) for the frame currently being shown, smoothed over that
    person's last RECENT_OBSERVATION_WINDOW sightings.

    An item shows as present if it is detected in THIS frame, or in at
    least two of the last few sightings. That balances three things:
      1. An item just put on shows up the moment it's first detected,
         rather than needing several frames of buildup (a majority vote
         needs 3 of 5 before it flips on, so a helmet plainly visible in
         this frame's own box could still read as missing).
      2. An item doesn't disappear just because ONE frame in between
         missed it (motion blur, a turned head, brief occlusion) even
         though the person never took it off.
      3. A single stray misdetection several frames ago no longer keeps
         an item lit up for the rest of the window — the earlier "seen in
         ANY recent frame" rule let exactly that happen, which is what
         made live detection look like it was inventing PPE.
    A false positive in the current frame still shows for that frame;
    that's the frame's own evidence, and it clears itself immediately.
    The whole-video verdict (_aggregate_tracked_summary) stays on the
    stricter majority vote regardless.
    """
    track = tracks.get(track_id)
    if not track or not track["observations"]:
        return None
    observations = track["observations"]
    recent = observations[-RECENT_OBSERVATION_WINDOW:]
    current = observations[-1]
    return {
        item: bool(current[item]) or sum(1 for o in recent if o[item]) >= 2
        for item in PPE_ITEMS
    }


# What it takes for the cumulative table to accept an item.
#
# The rule is "sustained presence", not a raw count: the item must appear at
# least CUMULATIVE_MIN_IN_WINDOW times inside some window of
# CUMULATIVE_WINDOW consecutive sightings of that person.
#
# A flat count doesn't work over a long video. Requiring 2 detections sounds
# strict until the track is 300 frames long — a class that misfires on 0.7%
# of frames still clears the bar and then stays ticked forever, which is
# exactly how a bare-handed worker ends up with gloves. Scaling the count
# with track length has the opposite failure: PPE put on in the last few
# seconds of a long video could never accumulate enough hits.
#
# Sustained presence sidesteps both. Real PPE, once on, is detected in most
# consecutive frames, so it clears the window almost immediately no matter
# how long the video is. Sporadic misfires are isolated by nature — they
# almost never land 3-in-5 — so they never confirm, however long you record.
CUMULATIVE_WINDOW = 5
CUMULATIVE_MIN_IN_WINDOW = 3


def _confirmed_observation(observations, item):
    """
    The observation at which `item` becomes confirmed for this track, or
    None if it never does.

    Confirmation means CUMULATIVE_MIN_IN_WINDOW hits inside a window of
    CUMULATIVE_WINDOW consecutive sightings — see the constants above for
    why sustained presence beats a raw count. The returned observation is
    the hit that completed the window, so the UI can report when the item
    was actually established rather than when it was first glimpsed.

    Tracks shorter than the window fall back to a simple majority of what
    little evidence exists, so a person who only appears in two or three
    frames isn't automatically stripped of everything they're wearing.
    """
    total = len(observations)
    if total == 0:
        return None

    if total < CUMULATIVE_WINDOW:
        needed = max(1, (total + 1) // 2)
        hits = [o for o in observations if o[item]]
        return hits[needed - 1] if len(hits) >= needed else None

    hits_in_window = 0
    for index, observation in enumerate(observations):
        if observation[item]:
            hits_in_window += 1
        # Drop the observation that just fell out of the trailing window.
        if index >= CUMULATIVE_WINDOW and observations[index - CUMULATIVE_WINDOW][item]:
            hits_in_window -= 1
        if hits_in_window >= CUMULATIVE_MIN_IN_WINDOW:
            return observation

    return None


def _cumulative_tracked_workers(tracks, fps=None):
    """
    One row per unique tracked person for the WHOLE video: an item counts
    as present if that person was seen wearing it in ANY analyzed frame,
    not just the frame currently on screen.

    This is what the Detection Result table shows. A per-frame table (even
    the recent-window smoothed one) answers "what is this person wearing at
    this instant", which reads as a bug when you're scrubbing: a worker who
    puts on a vest at 0.7s and a helmet at 3s never shows both ticked at
    once, so the table looks like it never notices the helmet. The question
    the table is actually being asked is "did this worker have their PPE on
    at any point in this clip" — so accumulate over the whole track.

    An item only counts once it has been detected in a sustained burst
    (see _confirmed_observation), not merely a couple of times across the
    whole clip — a tick that never expires has to be worth something, and
    over a few hundred frames "a couple of times" is indistinguishable from
    noise. Genuinely worn PPE clears the bar within about three frames of
    coming into view.

    Remaining trade-off: PPE that is taken off later still reads as present,
    because this view answers "did they have it on at any point", not "are
    they wearing it now". That's why the summary cards
    (_aggregate_tracked_summary) stay on the stricter recent-window majority
    vote — this is the per-worker checklist, not the compliance verdict, and
    the two can legitimately disagree.

    frames_seen / detection_counts / first_seen_sec are included so the UI
    can show how much evidence is behind each tick.
    """
    workers = []
    for position, (track_id, track) in enumerate(sorted(tracks.items()), start=1):
        observations = track["observations"]
        if not observations:
            continue

        worker = {
            "worker_id": position,
            "track_id": track_id,
            "frames_seen": len(observations),
            "first_seen_sec": None,
            "first_seen": {},
            "detection_counts": {},
        }

        for item in PPE_ITEMS:
            confirmed_at = _confirmed_observation(observations, item)
            worker[item] = confirmed_at is not None
            worker["detection_counts"][item] = sum(1 for o in observations if o[item])
            # The frame the item became *confirmed*, not the first stray
            # sighting — otherwise the timestamp points at evidence the
            # table itself didn't accept.
            worker["first_seen"][item] = (
                round(confirmed_at["frame_idx"] / fps, 2)
                if confirmed_at is not None and fps and confirmed_at.get("frame_idx") is not None
                else None
            )

        first_idx = observations[0].get("frame_idx")
        if fps and first_idx is not None:
            worker["first_seen_sec"] = round(first_idx / fps, 2)

        evaluate_worker(worker)
        workers.append(worker)

    return workers


@app.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    site_name: str = Form("Construction Site A"),
    current_user: dict = Depends(require_admin)
):
    """
    Runs the same YOLO -> PPE-association -> violation-check pipeline used
    for images, applied to every frame of an uploaded video — and streams
    each frame's result back to the client as soon as it's ready (newline-
    delimited JSON) so the UI can show live detection progress instead of a
    blank spinner until the whole video finishes.

    Safety caps (see MAX_VIDEO_FRAMES / MAX_VIDEO_RESULT_FRAMES near the top
    of this file): if a video is short enough, EVERY frame is processed. If
    it's longer than the cap, frames are sampled evenly across its full
    length so the whole video is still represented rather than just the
    first few seconds. Of the processed frames, only a capped number are
    persisted as annotated result images long-term (frames with violations
    are prioritized) to keep storage and the saved report reasonable — the
    live stream still shows every processed frame in the moment.
    """
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        }

    video_path = os.path.join(VIDEO_FOLDER, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open video file. It may be corrupted or use an unsupported codec."
        }

    def event_stream():
        start = time.perf_counter()

        # ------------------------------------------------------------
        # Rotation fix: phone videos shot in portrait store a rotation
        # flag in the container metadata (e.g. "rotate 90") rather than
        # actually storing sideways pixels. cv2.VideoCapture does NOT
        # apply this automatically on most builds, so without this fix
        # every frame handed to YOLO is sideways/upside-down, which tanks
        # detection accuracy across the board. We disable any built-in
        # auto-rotation (so behavior is consistent across OpenCV builds)
        # and apply the rotation ourselves based on the reported metadata.
        # ------------------------------------------------------------
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        try:
            orientation_deg = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
        except Exception:
            orientation_deg = 0
        rotate_flag = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(orientation_deg)

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / fps, 1) if fps and total_frames > 0 else None

        sample_every = (total_frames / MAX_VIDEO_FRAMES) if total_frames > MAX_VIDEO_FRAMES else 1.0
        frames_sampled = total_frames > MAX_VIDEO_FRAMES

        base_name = os.path.splitext(file.filename)[0]

        yield json.dumps({
            "type": "start",
            "frames_total": total_frames if total_frames > 0 else None,
            "duration_sec": duration_sec,
            "fps": round(fps, 2) if fps else None,
            "frames_sampled": frames_sampled,
        }) + "\n"

        def read_frame():
            """Reads the next frame and applies rotation correction, if any."""
            ok, raw = cap.read()
            if not ok:
                return None
            if rotate_flag is not None:
                raw = cv2.rotate(raw, rotate_flag)
            return raw

        def sharpness_score(img):
            """Higher = sharper / less motion-blurred (Laplacian variance)."""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()

        # ------------------------------------------------------------
        # Cross-frame person tracker — appearance (color-histogram) based,
        # not just position. Without this, the same worker seen across N
        # processed frames gets counted as N different "workers".
        #
        # Pure position/IoU matching (the first version of this) breaks
        # down on sampled long videos: if only every 5th-10th raw frame is
        # processed, a walking person's box can move too far between two
        # processed frames for IoU to catch — every track "expired" almost
        # immediately, which is why totals kept climbing every frame.
        #
        # Instead, each track keeps a running color-histogram "appearance"
        # signature (built from the pixels inside their bounding box —
        # mostly clothing/PPE color), and new detections are matched to the
        # most visually similar existing track regardless of how far apart
        # the frames are. Bounding-box overlap is still used as a *bonus*
        # signal when two processed frames happen to be close together, to
        # help disambiguate multiple similarly-dressed workers.
        #
        # Honest limitation: this is still not face recognition. Workers in
        # near-identical PPE/uniforms can look alike to a color histogram
        # and may occasionally get merged or split. It's a solid, dependency
        # -free improvement over pure position tracking, not perfect ReID.
        # ------------------------------------------------------------
        APPEARANCE_MATCH_THRESHOLD = 0.5
        SPATIAL_BONUS_FRAME_WINDOW = 30  # raw video frames (~1s at 30fps) within which IoU is trusted as a bonus signal
        # Box overlap this high, within the window above, is treated as proof
        # of continuity on its own — see the note in assign_tracks about
        # donning PPE wrecking a person's color signature.
        STRONG_IOU_CONTINUITY = 0.35
        tracks = {}
        next_track_id = [1]

        def person_histogram(frame_img, box):
            h, w = frame_img.shape[:2]
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame_img[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist

        def histogram_similarity(hist_a, hist_b):
            if hist_a is None or hist_b is None:
                return 0.0
            return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

        def assign_tracks(frame_idx, workers, frame_img):
            # Crowded-frame handling: score every (detected person, existing
            # track) pair up front, then commit matches highest-score-first
            # across the WHOLE frame rather than letting whichever person
            # the model happened to list first grab its best track before
            # anyone else gets a say. That row-by-row version worked fine
            # for one or two people but could misassign in a busy frame —
            # e.g. person A is a great match for track 1 and an OK match for
            # track 2, person B is a great match for track 1 and a poor
            # match for anything else; if A were processed first it would
            # simply take track 1 and leave B stranded, even though A also
            # had a perfectly good second option. Sorting all candidate
            # pairs by score first resolves that kind of conflict correctly.
            hists = [person_histogram(frame_img, w["person_box"]) for w in workers]

            candidates = []
            for wi, (w, hist) in enumerate(zip(workers, hists)):
                box = w["person_box"]
                for tid, t in tracks.items():
                    appearance_score = histogram_similarity(hist, t["hist"])
                    nearby_in_time = (frame_idx - t["last_frame_idx"]) <= SPATIAL_BONUS_FRAME_WINDOW
                    iou_score = _box_iou(box, t["box"]) if nearby_in_time else 0.0
                    score = appearance_score * 0.75 + iou_score * 0.25

                    # Putting PPE ON is exactly the event that breaks an
                    # appearance-only match: pulling a hi-vis vest over a
                    # blue shirt rewrites most of the color histogram in one
                    # frame, the similarity collapses, and the person is
                    # handed a brand-new track id. That's what makes PPE
                    # "stop accumulating" — the helmet detected earlier is
                    # stranded on the old track while everything donned
                    # afterwards lands on a new one, so no single worker row
                    # ever shows the full set. A person cannot jump across
                    # the frame between two nearby frames, so strong box
                    # overlap in that window is treated as continuity in its
                    # own right rather than a mere tie-breaker.
                    if nearby_in_time and iou_score >= STRONG_IOU_CONTINUITY:
                        score = max(score, APPEARANCE_MATCH_THRESHOLD + iou_score * 0.25)

                    if score >= APPEARANCE_MATCH_THRESHOLD:
                        candidates.append((score, wi, tid))

            candidates.sort(key=lambda c: c[0], reverse=True)

            assigned_track = {}
            claimed_tracks = set()
            claimed_workers = set()
            for score, wi, tid in candidates:
                if wi in claimed_workers or tid in claimed_tracks:
                    continue
                assigned_track[wi] = tid
                claimed_workers.add(wi)
                claimed_tracks.add(tid)

            # Single-person fallback: one person on screen and exactly one
            # recently-seen track means there is nobody else they could be.
            # Insisting on a color match there can only ever split one
            # worker into several — which is the common case for these
            # clips (one person demonstrating PPE) and the worst case for
            # the cumulative table.
            if len(workers) == 1 and 0 not in assigned_track:
                recent = [
                    tid for tid, t in tracks.items()
                    if (frame_idx - t["last_frame_idx"]) <= SPATIAL_BONUS_FRAME_WINDOW
                ]
                if len(recent) == 1:
                    assigned_track[0] = recent[0]

            for wi, w in enumerate(workers):
                box = w["person_box"]
                hist = hists[wi]

                if wi in assigned_track:
                    tid = assigned_track[wi]
                    # Ease the stored signature toward the new observation so
                    # it can drift gracefully with lighting/pose over time.
                    if hist is not None:
                        if tracks[tid]["hist"] is None:
                            tracks[tid]["hist"] = hist
                        else:
                            tracks[tid]["hist"] = cv2.addWeighted(tracks[tid]["hist"], 0.7, hist, 0.3, 0)
                else:
                    tid = next_track_id[0]
                    next_track_id[0] += 1
                    tracks[tid] = {"hist": hist, "box": box, "last_frame_idx": frame_idx, "observations": []}

                tracks[tid]["box"] = box
                tracks[tid]["last_frame_idx"] = frame_idx
                observation = {item: w[item] for item in PPE_ITEMS}
                # Kept alongside the PPE flags so the cumulative whole-video
                # table can report *when* each item was first seen.
                observation["frame_idx"] = frame_idx
                tracks[tid]["observations"].append(observation)
                w["track_id"] = tid

        all_frame_results = []
        next_target = 0.0
        frame_cursor = 0
        processed_count = 0

        try:
            while True:
                frame = read_frame()
                if frame is None:
                    break
                idx = frame_cursor
                frame_cursor += 1

                # Unknown-length video: process every frame we see, but never
                # exceed the hard cap so a weird/corrupt video can't run forever.
                if total_frames <= 0 and processed_count >= MAX_VIDEO_FRAMES:
                    break

                if idx < next_target:
                    continue

                next_target += sample_every

                # For sampled (long) videos, peek ahead a few frames and keep
                # the sharpest one instead of blindly using the first frame at
                # the sample point. Motion blur is a major cause of missed or
                # incorrect PPE detections in video, and this costs nothing on
                # short "every frame" videos since the window is skipped there.
                best_idx, best_frame = idx, frame
                if frames_sampled:
                    best_sharpness = sharpness_score(frame)
                    window = min(BLUR_SEARCH_WINDOW, max(0, int(sample_every) - 1))
                    for _ in range(window):
                        peek = read_frame()
                        if peek is None:
                            break
                        peek_idx = frame_cursor
                        frame_cursor += 1
                        peek_sharpness = sharpness_score(peek)
                        if peek_sharpness > best_sharpness:
                            best_idx, best_frame, best_sharpness = peek_idx, peek, peek_sharpness

                frame_name = f"{base_name}_frame{best_idx:05d}.jpg"
                frame_path = os.path.join(UPLOAD_FOLDER, frame_name)
                cv2.imwrite(frame_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                try:
                    results = detect_objects(frame_path)

                    output_path = os.path.join(OUTPUT_FOLDER, frame_name)
                    results[0].save(filename=output_path)

                    workers = associate_ppe(results, keep_person_box=True)
                    report = check_violations(workers)
                    processed_workers = report["workers"]

                    # Link each detected person to a stable track_id before
                    # dropping their pixel box (which the frontend doesn't
                    # need) — this is what turns "N workers per frame" into
                    # "N unique workers across the whole video".
                    assign_tracks(best_idx, processed_workers, best_frame)
                    for w in processed_workers:
                        w.pop("person_box", None)

                    # Replace each tracked worker's raw single-frame PPE
                    # flags with the recent-window smoothed version (see
                    # _smoothed_worker_ppe) — otherwise the "workers" table
                    # shown for THIS frame is just whatever YOLO happened to
                    # catch in this one instant, so an item detected clearly
                    # a few frames ago (or a few frames from now) can drop
                    # out of view even though the person never took it off.
                    # Re-run check_violations afterward so every derived
                    # number (missing list, status, safe/unsafe counts,
                    # compliance %) matches the smoothed flags instead of
                    # the pre-smoothing raw ones.
                    for w in processed_workers:
                        smoothed = _smoothed_worker_ppe(w.get("track_id"), tracks)
                        if smoothed:
                            w.update(smoothed)
                    report = check_violations(processed_workers)
                    processed_workers = report["workers"]

                    image_safe = report["summary"]["safe_workers"]
                    image_unsafe = report["summary"]["unsafe_workers"]
                    image_total = image_safe + image_unsafe
                    # % of required PPE items present, matching
                    # compute_compliance_rate everywhere else — not
                    # safe/total workers (see api.py /detect for why).
                    image_compliance = compute_compliance_rate(processed_workers)

                    missing_ppe = []
                    for worker in processed_workers:
                        missing_ppe.extend(worker.get("missing", []))
                    missing_ppe = list(set(missing_ppe))

                    timestamp_sec = round(best_idx / fps, 2) if fps else None

                    frame_result = {
                        "name": frame_name,
                        "site_name": site_name,
                        "original": f"http://127.0.0.1:8000/uploads/{frame_name}",
                        "annotated": f"http://127.0.0.1:8000/outputs/{frame_name}",
                        "workers": processed_workers,
                        "safe_workers": image_safe,
                        "unsafe_workers": image_unsafe,
                        "total_workers": image_total,
                        "compliance_rate": image_compliance,
                        "missing": missing_ppe,
                        "status": "Safe" if image_unsafe == 0 else "Unsafe",
                        "frame_index": best_idx,
                        "timestamp_sec": timestamp_sec,
                    }
                    all_frame_results.append(frame_result)
                    processed_count += 1

                    # Live progress event — the frontend renders this frame
                    # immediately, plus running totals based on unique people
                    # tracked so far (not raw per-frame sums) so the counters
                    # climbing in real time actually mean something.
                    yield json.dumps({
                        "type": "frame",
                        "processed_count": processed_count,
                        "frames_total": total_frames if total_frames > 0 else None,
                        "summary_so_far": _aggregate_tracked_summary(tracks, processed_count, fps),
                        # Per-worker checklist accumulated over every frame
                        # analyzed so far, so the live table keeps items
                        # ticked once they've been detected instead of
                        # resetting to whatever this single frame caught.
                        "workers_cumulative": _cumulative_tracked_workers(tracks, fps),
                        **frame_result,
                    }) + "\n"
                except Exception as frame_error:
                    print(f"⚠️ Frame {best_idx} detection failed: {frame_error}")
        finally:
            cap.release()

        # Final summary based on unique tracked people across the whole
        # video (see _aggregate_tracked_summary) rather than raw per-frame
        # detection sums.
        summary = _aggregate_tracked_summary(tracks, processed_count, fps)

        # Whole-video per-worker checklist. Computed from every processed
        # frame's observations, not just the frames kept below — dropping a
        # frame's image file shouldn't lose the PPE it showed.
        cumulative_workers = _cumulative_tracked_workers(tracks, fps)

        # ----------------------------------------------------
        # Keep only a capped set of frames as annotated results:
        # violations first, then evenly spaced safe frames to fill the rest.
        # ----------------------------------------------------
        violation_frames = [f for f in all_frame_results if f["status"] == "Unsafe"]
        safe_frames = [f for f in all_frame_results if f["status"] == "Safe"]

        kept = violation_frames[:MAX_VIDEO_RESULT_FRAMES]
        remaining_slots = MAX_VIDEO_RESULT_FRAMES - len(kept)
        if remaining_slots > 0 and safe_frames:
            step = max(1, len(safe_frames) // remaining_slots)
            kept += safe_frames[::step][:remaining_slots]

        kept_names = {f["name"] for f in kept}

        # Remove on-disk frame images that didn't make the cut, to save
        # space. Safe to do only now — every frame has already been
        # streamed live to the client by this point.
        for f in all_frame_results:
            if f["name"] not in kept_names:
                for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
                    path = os.path.join(folder, f["name"])
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass

        kept.sort(key=lambda f: f["frame_index"])

        processing_time = round(time.perf_counter() - start, 2)

        # ----------------------------------------------------
        # Save Detection History to MongoDB
        # (uses the same schema as image detections — `images` holds the kept
        # result frames — so the existing dashboard/detail views work unchanged)
        # ----------------------------------------------------
        record = {
            "timestamp": str(datetime.now()),
            "summary": summary,
            "images": kept,
            "workers_cumulative": cumulative_workers,
            "source_type": "video",
            "video_filename": file.filename,
            "video_url": f"http://127.0.0.1:8000/videos/{file.filename}",
            "fps": round(fps, 2) if fps else None,
            "duration_sec": duration_sec,
            "frames_total": total_frames if total_frames > 0 else processed_count,
            "frames_processed": processed_count,
            "frames_sampled": frames_sampled,
            "acknowledged": False,
        }

        try:
            detections.insert_one(record)
            print("Video detection history saved to MongoDB.")
        except Exception as e:
            print(f"MongoDB Error: {e}")

        yield json.dumps({
            "type": "done",
            "success": True,
            "summary": summary,
            "images": kept,
            "workers_cumulative": cumulative_workers,
            "frames_total": total_frames if total_frames > 0 else processed_count,
            "frames_processed": processed_count,
            "frames_sampled": frames_sampled,
            "duration_sec": duration_sec,
            "processing_time": processing_time,
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ----------------------------------------------------
# Download PDF Report Endpoint
# ----------------------------------------------------
@app.get("/download-report/{inspection_id}")
def download_report(inspection_id: str, current_user: dict = Depends(get_current_user)):
    """
    Download a PDF report for a specific inspection.
    """
    try:
        # Find the inspection record
        inspection = detections.find_one(
            {
                "_id": ObjectId(inspection_id)
            }
        )

        if inspection is None:
            raise HTTPException(
                status_code=404,
                detail="Inspection not found."
            )

        # Self-heal older records the same way _serialize_detection does —
        # recompute each image's compliance_rate from its own worker data
        # so the PDF doesn't show the old binary 0%/100% formula.
        for image in inspection.get("images", []):
            if image.get("workers"):
                image["compliance_rate"] = compute_compliance_rate(image["workers"])

        # Generate PDF path
        pdf_path = os.path.join(
            REPORT_FOLDER,
            f"{inspection_id}.pdf"
        )

        # Generate the PDF report
        generate_pdf_report(
            inspection,
            pdf_path
        )

        # Return the PDF file
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"Inspection_{inspection_id}.pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ----------------------------------------------------
# Detection History: List / Detail / Delete
# ----------------------------------------------------
def _serialize_detection(doc, include_images=True):
    images = doc.get("images", [])
    sites = sorted({
        img.get("site_name") for img in images if img.get("site_name")
    })

    # Self-heal older records: their stored per-image compliance_rate may
    # have used the old safe/total-workers formula (binary 0%/100% for a
    # single-worker image) instead of compute_compliance_rate's items-
    # present formula. Recompute from each image's own worker data on read
    # so this is correct everywhere without a database migration.
    if include_images:
        images = [
            {**img, "compliance_rate": compute_compliance_rate(img["workers"])}
            if img.get("workers")
            else img
            for img in images
        ]

    item = {
        "id": str(doc["_id"]),
        "timestamp": doc.get("timestamp"),
        "summary": doc.get("summary", {}),
        "sites": sites,
        # Records saved before this field existed were always image uploads.
        "source_type": doc.get("source_type", "image"),
        "video_filename": doc.get("video_filename"),
        "duration_sec": doc.get("duration_sec"),
        "acknowledged": doc.get("acknowledged", False),
    }
    if include_images:
        item["images"] = images
        # Whole-video per-tracked-worker checklist (video records only, and
        # only those saved after this field was added) — the cumulative view
        # the upload screen shows, kept available to detail views too.
        if doc.get("workers_cumulative"):
            item["workers_cumulative"] = doc["workers_cumulative"]
    return item


@app.get("/detections")
def list_detections(current_user: dict = Depends(get_current_user)):
    """
    List all past inspections (most recent first).
    Used by the read-only user dashboard and the admin 'Manage Detections' page.
    """
    try:
        docs = list(detections.find().sort("timestamp", -1))
        return {
            "success": True,
            "count": len(docs),
            "detections": [_serialize_detection(d, include_images=False) for d in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detections/{detection_id}")
def get_detection(detection_id: str, current_user: dict = Depends(get_current_user)):
    try:
        doc = detections.find_one({"_id": ObjectId(detection_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid detection id.")

    if doc is None:
        raise HTTPException(status_code=404, detail="Detection not found.")

    return {
        "success": True,
        "detection": _serialize_detection(doc, include_images=True)
    }


@app.delete("/detections/{detection_id}")
def delete_detection(detection_id: str, current_user: dict = Depends(require_admin)):
    try:
        result = detections.delete_one({"_id": ObjectId(detection_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid detection id.")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Detection not found.")

    return {"success": True, "message": "Detection deleted."}


# ----------------------------------------------------
# Violation Notifications
# ----------------------------------------------------
NOTIFICATION_QUERY = {
    "summary.unsafe_workers": {"$gt": 0},
    "acknowledged": {"$ne": True},
}


@app.get("/notifications")
def list_notifications(current_user: dict = Depends(get_current_user)):
    """
    Recent inspections that flagged at least one PPE violation and haven't
    been acknowledged yet — powers the notification bell in the topbar.
    """
    docs = list(
        detections.find(NOTIFICATION_QUERY).sort("timestamp", -1).limit(20)
    )
    unread_count = detections.count_documents(NOTIFICATION_QUERY)

    items = []
    for d in docs:
        summary = d.get("summary", {}) or {}
        items.append({
            "id": str(d["_id"]),
            "timestamp": d.get("timestamp"),
            "source_type": d.get("source_type", "image"),
            "unsafe_workers": summary.get("unsafe_workers", 0),
            "total_workers": summary.get("total_workers", 0),
            "compliance": summary.get("compliance", 0),
            "sites": sorted({
                img.get("site_name") for img in d.get("images", []) if img.get("site_name")
            }),
        })

    return {
        "success": True,
        "unread_count": unread_count,
        "notifications": items,
    }


@app.post("/notifications/{detection_id}/ack")
def acknowledge_notification(detection_id: str, current_user: dict = Depends(get_current_user)):
    try:
        result = detections.update_one(
            {"_id": ObjectId(detection_id)},
            {"$set": {"acknowledged": True}},
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid detection id.")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Detection not found.")

    return {"success": True}


@app.post("/notifications/read-all")
def acknowledge_all_notifications(current_user: dict = Depends(get_current_user)):
    detections.update_many(NOTIFICATION_QUERY, {"$set": {"acknowledged": True}})
    return {"success": True}


# ----------------------------------------------------
# Upload Safety Manual / Inspection Report
# ----------------------------------------------------
@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_admin)
):
    start = time.perf_counter()

    try:
        print("=" * 60)
        print("📤 DOCUMENT UPLOAD REQUEST RECEIVED")
        print(f"   Filename: {file.filename}")
        print(f"   Content-Type: {file.content_type}")
        print("=" * 60)

        print("1️⃣ Checking file extension...")
        # Check if file extension is supported
        allowed_extensions = [".pdf", ".docx", ".txt"]
        extension = os.path.splitext(file.filename)[1].lower()

        if extension not in allowed_extensions:
            print(f"❌ Unsupported extension: {extension}")
            return {
                "success": False,
                "error": "Only PDF, DOCX and TXT files are supported."
            }
        print(f"   ✅ Extension '{extension}' is supported.")

        # Determine document type based on extension
        if extension == ".pdf":
            doc_type = "PDF Manual"
        elif extension == ".docx":
            doc_type = "DOCX Manual"
        elif extension == ".txt":
            doc_type = "Text Document"
        else:
            doc_type = "Unknown"
        print(f"   📄 Document type: {doc_type}")

        # Save uploaded file
        print("2️⃣ Saving file...")
        save_path = os.path.join(DOCUMENT_FOLDER, file.filename)
        print(f"   📁 Path: {save_path}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("   ✅ File saved successfully.")

        # Extract text
        print("3️⃣ Extracting text...")
        extracted_text = extract_text(save_path)
        print(f"   📝 Text extracted: {len(extracted_text)} characters")
        print(f"   📊 Preview: {extracted_text[:100]}...")

        # Store in MongoDB with document type
        print("4️⃣ Storing in MongoDB...")
        document = {
            "filename": file.filename,
            "filepath": save_path,
            "document_type": doc_type,
            "uploaded_at": str(datetime.now()),
            "text": extracted_text
        }

        # Replace existing document if filename already exists, otherwise insert
        result = documents.replace_one(
            {"filename": file.filename},
            document,
            upsert=True
        )
        print(f"   ✅ MongoDB updated. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted: {result.upserted_id is not None}")

        # ----------------------------------------------------
        # ⚠️ SYNC KNOWLEDGE BASE UPDATE (College Project)
        # ----------------------------------------------------
        print("5️⃣ Updating knowledge base...")
        kb_updated = True
        try:
            update_knowledge_base()
            print(f"   ✅ Knowledge base updated with {file.filename}")
        except Exception as e:
            kb_updated = False
            print(f"   ❌ Knowledge base update error: {e}")

        # Calculate processing time
        processing_time = round(
            time.perf_counter() - start,
            2
        )

        # ----------------------------------------------------
        # Return Response with Knowledge Base Status
        # ----------------------------------------------------
        print("=" * 60)
        print("✅ UPLOAD COMPLETE")
        print(f"   Knowledge Base Updated: {kb_updated}")
        print(f"   Processing Time: {processing_time}s")
        print("=" * 60)

        word_count = len(extracted_text.split())
        
        return {
            "success": True,
            "message": "Document uploaded successfully.",
            "knowledge_base_updated": kb_updated,
            "filename": file.filename,
            "document_type": doc_type,
            "characters": len(extracted_text),
            "word_count": word_count,
            "url": f"http://127.0.0.1:8000/documents/{file.filename}",
            "processing_time": processing_time,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "FAISS",
            "llm": "Llama 3.1"
        }

    except Exception as e:
        print("=" * 60)
        print("❌ UPLOAD FAILED")
        print(f"   Error: {str(e)}")
        print("=" * 60)
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }


# ----------------------------------------------------
# Document Management: List / Delete (Admin)
# ----------------------------------------------------
@app.get("/documents")
def list_documents(current_user: dict = Depends(require_admin)):
    """
    Full metadata list of uploaded RAG documents/manuals for the
    admin 'Manage Documents' page (excludes the large extracted text body).
    """
    try:
        docs = list(
            documents.find(
                {},
                {
                    "_id": 0,
                    "text": 0
                }
            )
        )
        return {
            "success": True,
            "count": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
def delete_document(filename: str, current_user: dict = Depends(require_admin)):
    doc = documents.find_one({"filename": filename})

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    filepath = doc.get("filepath")
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"⚠️ Could not remove file '{filepath}': {e}")

    documents.delete_one({"filename": filename})

    kb_updated = True
    try:
        update_knowledge_base()
    except Exception as e:
        kb_updated = False
        print(f"❌ Knowledge base update error after delete: {e}")

    return {
        "success": True,
        "message": f"'{filename}' deleted.",
        "knowledge_base_updated": kb_updated
    }


# ----------------------------------------------------
# Analytics (Admin only)
# ----------------------------------------------------
@app.get("/analytics")
def get_analytics(current_user: dict = Depends(require_admin)):
    """
    Aggregates the detection_history collection into:
      - summary totals
      - PPE violation breakdown
      - a compliance trend over time (grouped by day)
    """
    try:
        docs = list(detections.find().sort("timestamp", 1))

        totals = {
            "total_inspections": len(docs),
            "total_images": 0,
            "total_workers": 0,
            "safe_workers": 0,
            "unsafe_workers": 0,
        }

        violation_breakdown = {
            "helmet": 0,
            "vest": 0,
            "gloves": 0,
            "boots": 0,
            "goggles": 0,
        }

        trend_by_day = {}

        for doc in docs:
            summary = doc.get("summary", {}) or {}

            totals["total_images"] += summary.get("total_images", 0)
            totals["total_workers"] += summary.get("total_workers", 0)
            totals["safe_workers"] += summary.get("safe_workers", 0)
            totals["unsafe_workers"] += summary.get("unsafe_workers", 0)

            for key in violation_breakdown:
                violation_breakdown[key] += summary.get(key, 0)

            timestamp = doc.get("timestamp", "")
            day = timestamp[:10] if timestamp else "unknown"

            bucket = trend_by_day.setdefault(
                day,
                {"date": day, "inspections": 0, "safe": 0, "unsafe": 0, "compliance_sum": 0.0}
            )
            bucket["inspections"] += 1
            bucket["safe"] += summary.get("safe_workers", 0)
            bucket["unsafe"] += summary.get("unsafe_workers", 0)
            bucket["compliance_sum"] += summary.get("compliance", 0) or 0

        trend = []
        for day in sorted(trend_by_day.keys()):
            bucket = trend_by_day[day]
            avg_compliance = round(bucket["compliance_sum"] / bucket["inspections"], 2) if bucket["inspections"] else 0
            trend.append({
                "date": bucket["date"],
                "inspections": bucket["inspections"],
                "safe": bucket["safe"],
                "unsafe": bucket["unsafe"],
                "compliance": avg_compliance,
            })

        overall_required = totals["total_workers"] * 5
        overall_missing = sum(violation_breakdown.values())
        avg_compliance = (
            round(((overall_required - overall_missing) / overall_required) * 100, 2)
            if overall_required > 0 else 100.0
        )

        return {
            "success": True,
            "totals": totals,
            "avg_compliance": avg_compliance,
            "violation_breakdown": violation_breakdown,
            "trend": trend,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# AI Assistant (RAG Chat) - LangGraph Integration
# ----------------------------------------------------
def _friendly_llm_error(raw_message):
    """
    Translate the chat model's raw exception text into something a user can
    act on.

    Which failures are even possible depends on where the model runs, so
    both sets are handled. Locally, Ollama's llama-server process can crash
    outright on GPU/driver faults ("exit status 0xc0000409", "CUDA error:
    ..."). Against a hosted provider those never occur; instead you get
    rejected keys, rate limits and timeouts — which, before this, fell
    through as raw Python exception strings.
    """
    lowered = raw_message.lower()

    if "llama-server" in lowered or "cuda" in lowered or "exit status" in lowered:
        return (
            "The local AI model (Ollama) crashed while generating a response — this is "
            "usually a GPU/CUDA driver problem on the machine running Ollama, not an issue "
            "with Safesight itself. We automatically retried once. If it keeps happening: "
            "restart the Ollama service, update your NVIDIA/GPU drivers, or run Ollama in "
            "CPU mode (set the OLLAMA_LLM_LIBRARY=cpu environment variable) and restart it."
        )

    if "connection" in lowered and ("refused" in lowered or "failed" in lowered):
        return (
            "Couldn't reach the AI model. If you're running locally, check that Ollama is "
            "started; if you're using a hosted provider, check this machine's internet "
            "connection."
        )

    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return (
            "The AI provider's rate limit was hit — too many questions in a short window. "
            "Wait a moment and ask again. On a free tier this is normal during a burst of "
            "questions; the limits reset every minute."
        )

    if "401" in lowered or "403" in lowered or "api key" in lowered or "unauthorized" in lowered:
        return (
            "The AI provider rejected the API key. Check that the key environment variable "
            "is set correctly for the configured provider and that the key is still active, "
            "then restart the backend."
        )

    if "timeout" in lowered or "timed out" in lowered:
        return (
            "The AI model took too long to respond. Try a shorter question, or lower "
            "SAFESIGHT_LLM_NUM_PREDICT to cap answer length."
        )

    if "not found" in lowered and "model" in lowered:
        return (
            "The configured model name wasn't recognised by the provider. Model names change "
            "regularly — check the provider's current model list and set SAFESIGHT_LLM_MODEL "
            "accordingly (locally, make sure you've run `ollama pull <model>`)."
        )

    return raw_message


CHAT_MAX_RETRIES = 1


@app.post("/chat")
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    start = time.perf_counter()

    try:
        print("=" * 60)
        print("💬 CHAT REQUEST RECEIVED")
        print(f"   Question: {request.question}")
        print(f"   Search Scope: {request.search_scope}")
        if request.selected_manuals:
            print(f"   Selected Manuals: {', '.join(request.selected_manuals)}")
        print("=" * 60)

        # Validate that question is not empty
        if not request.question or not request.question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty. Please provide a valid question."
            }

        # ✅ Step 1: Invoke the LangGraph workflow — retried once, since a
        # crashed local llama-server process is usually back up (Ollama
        # respawns it) by the time a second request comes in a moment later.
        workflow_input = {
            "question": request.question,
            "search_scope": request.search_scope,
            "selected_manuals": request.selected_manuals,
            "mode": "",
            "results": [],
            "reports": [],
            "answer": "",
            "sources": [],
            "source_names": [],
            "summary": {},
            "has_inspections": False,
            "has_manuals": False,
            "evidence": {},
            "metrics": {},
            "is_summary_request": False,
            "manual_metadata": []
        }

        result = None
        for attempt in range(CHAT_MAX_RETRIES + 1):
            try:
                result = workflow.invoke(workflow_input)
                break
            except Exception as llm_error:
                print(f"⚠️ LangGraph invoke failed (attempt {attempt + 1}/{CHAT_MAX_RETRIES + 1}): {llm_error}")
                if attempt >= CHAT_MAX_RETRIES:
                    raise
                time.sleep(2)  # give Ollama a moment to respawn llama-server

        response_time = round(
            time.perf_counter() - start,
            2
        )

        # ✅ Step 2: Extract response from the workflow result
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        mode = result.get("mode", "unknown")
        source_count = len(sources)

        print("✅ Chat response generated successfully")
        print(f"   Answer length: {len(answer)} characters")
        print(f"   Sources: {source_count}")
        print(f"   Mode: {mode}")

        print(
            f"[CHAT] "
            f"Mode={mode} | "
            f"Sources={source_count} | "
            f"Time={response_time}s"
        )

        # ✅ Step 3: Build response
        response = {
            "success": True,
            "mode": mode,
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "source_count": source_count,
            "response_time": response_time
        }

        # Include summary if available (for analytics mode)
        if "summary" in result and result["summary"]:
            response["summary"] = result["summary"]

        return response

    except Exception as e:
        print("=" * 60)
        print("❌ CHAT FAILED")
        print(f"   Error: {str(e)}")
        print("=" * 60)
        traceback.print_exc()

        return {
            "success": False,
            "error": _friendly_llm_error(str(e))
        }


# ----------------------------------------------------
# Optional: Manual Knowledge Base Update Endpoint
# ----------------------------------------------------
@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base(current_user: dict = Depends(require_admin)):
    """
    Manually trigger a rebuild of the knowledge base.
    Useful if documents were added directly to MongoDB or filesystem.
    """
    try:
        update_knowledge_base()
        return {
            "success": True,
            "message": "Knowledge base rebuilt successfully."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild knowledge base: {str(e)}"
        )


# ----------------------------------------------------
# Test Endpoint: Check LangGraph Branch
# ----------------------------------------------------
@app.get("/test-intent")
def test_intent(question: str):
    """
    Test endpoint to check which LangGraph branch a question would take.
    """
    from langgraph_pipeline import intent_node
    
    state = {
        "question": question,
        "search_scope": "all",
        "selected_manuals": [],
        "mode": "",
        "results": [],
        "reports": [],
        "answer": "",
        "sources": [],
        "source_names": [],
        "summary": {},
        "has_inspections": False,
        "has_manuals": False
    }
    
    result = intent_node(state)
    
    return {
        "question": question,
        "mode": result.get("mode", "unknown")
    }