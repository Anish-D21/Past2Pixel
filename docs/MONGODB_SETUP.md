# Past2Pixel — MongoDB Atlas Persistence & Local Storage Guide

Past2Pixel utilizes **MongoDB Atlas** (Free Tier M0 cluster) for persistent document storage of user credentials, processing session metadata, OCR transcriptions, and quantitative pipeline results.

---

## 1. Cloud Setup: MongoDB Atlas (Free Tier M0)

### Step 1: Create a Free Cluster
1. Sign up or log in at [MongoDB Cloud](https://cloud.mongodb.com/).
2. Click **Build a Database** and select **M0 (Free)**.
3. Choose a cloud provider and region closest to your location (e.g., AWS / Mumbai or Frankfurt).
4. Name your cluster (e.g., `Cluster0` or `Past2PixelCluster`) and click **Create**.

### Step 2: Configure Database User Credentials
1. In the **Security > Database Access** tab:
   - Click **Add New Database User**.
   - Authentication Method: **Password**.
   - Enter a username (e.g., `past2pixel_admin`) and a strong password.
   - Database User Privileges: **Read and write to any database**.
   - Click **Add User**.

### Step 3: Network Access (IP Whitelisting)
1. In the **Security > Network Access** tab:
   - Click **Add IP Address**.
   - For local development, choose **Add Current IP Address** or **Allow Access from Anywhere** (`0.0.0.0/0`) if using a dynamic ISP.
   - Click **Confirm**.

### Step 4: Retrieve Connection String
1. In the **Deployment > Database** tab, click **Connect** next to your cluster.
2. Select **Drivers** (Python / PyMongo).
3. Copy the SRV connection URI:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
4. Replace `<username>` and `<password>` with your database user credentials.

### Step 5: Configure Past2Pixel Environment
In your `.env` file (copied from `.env.example`), set:
```env
MONGODB_URI=mongodb+srv://past2pixel_admin:YourSecurePassword123@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=past2pixel
```

---

## 2. Transparent Local Persistence Fallback

If `MONGODB_URI` is left blank, or if the remote MongoDB Atlas cluster is unreachable due to network restrictions or offline environments:
- **Automatic Fallback:** Past2Pixel switches seamlessly to **Local File Persistence**.
- **Storage Location:** `data/sessions/local_db.json`.
- **Honest Status:** The system explicitly tags all session records with `persistence_info: {"storage_mode": "local_file_persistence"}`.
- **Zero Data Loss:** Local registration, login, and session history remain fully functional and persistent across server restarts.

---

## 3. Database Schema Design

To prevent database bloat, **bulky high-resolution image binaries are never stored inside MongoDB documents**. Original, restored, and overlay images are persisted to the filesystem (`data/sessions/{session_id}/`), and lightweight metadata references are stored in MongoDB.

### Collections & Indexes

#### 1. `users` Collection
Stores user profiles and bcrypt password hashes.
```json
{
  "_id": "67155...",
  "email": "student@university.edu",
  "username": "aiml_student",
  "password_hash": "$2b$12$K...hashed_string...",
  "created_at": "2026-10-20T16:44:07.000000+00:00"
}
```
*Indexes:*
- `email`: Unique index
- `username`: Unique index

#### 2. `sessions` Collection
Stores image-specific restoration session records.
```json
{
  "_id": "67155...",
  "session_id": "p2p_79ce119fea00_1792515025",
  "user_id": "usr_0001_1792515025",
  "status": "RESTORATION_COMPLETED",
  "created_at": "2026-10-20T16:45:00.000000+00:00",
  "updated_at": "2026-10-20T16:45:15.000000+00:00",
  "original_metadata": {
    "original_filename": "vedic_folio.png",
    "width": 1606,
    "height": 146,
    "format": "PNG",
    "file_size_bytes": 142080
  },
  "artifacts": {
    "original": "/static/sessions/p2p_79ce119fea00_1792515025/original.png",
    "restored": "/static/sessions/p2p_79ce119fea00_1792515025/restored_unet.png",
    "damage_overlay": "/static/sessions/p2p_79ce119fea00_1792515025/damage_overlay.png",
    "damage_mask": "/static/sessions/p2p_79ce119fea00_1792515025/damage_mask.png"
  },
  "ocr_result": { ... },
  "restoration_result": { ... },
  "text_predictions": [ ... ],
  "persistence_info": {
    "storage_mode": "mongodb_atlas",
    "status": "SAVED",
    "saved_at": "2026-10-20T16:45:15.000000+00:00"
  }
}
```
*Indexes:*
- `session_id`: Unique index
- `user_id`: Standard index
- `created_at`: Standard index (descending)

---

## 4. Diagnostics & Safe Data Migration

### Checking Connection Status
Run the diagnostic command from any directory:
```powershell
python -c "from src.db.mongodb import db_manager; print(db_manager.get_status())"
```
When Atlas is connected:
```json
{
  "status": "CONNECTED",
  "storage_mode": "mongodb_atlas",
  "uri_configured": true,
  "database_name": "past2pixel",
  "is_persistent": true,
  "last_error": null,
  "local_store_path": null
}
```

### Safe Migration from Local JSON to Atlas
If you developed offline or previously used `data/sessions/local_db.json`, you can safely transfer your historical users and sessions to MongoDB Atlas:

1. **Audit with Dry Run (Zero Writes):**
   ```powershell
   python scripts/migrate_local_to_atlas.py --dry-run
   ```
2. **Execute Live Migration (Idempotent):**
   ```powershell
   python scripts/migrate_local_to_atlas.py --confirm
   ```
*Guarantees: Skips duplicate users and existing sessions; leaves `data/sessions/local_db.json` untouched as a backup.*

