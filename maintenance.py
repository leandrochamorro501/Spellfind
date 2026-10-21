"""Local quotas, reviewed retention plans and verified offline backups."""
import argparse
from contextlib import contextmanager, closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from uuid import uuid4
import zipfile

HEX = re.compile(r"[a-f0-9]{32}")


class StorageLimit(RuntimeError): pass


def safe_path(root, relative):
    root = Path(root).absolute()
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()) or target.resolve() == root.resolve():
        raise ValueError("Maintenance target escapes its managed root")
    for path in [root, *target.parents, target]:
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Maintenance refuses symbolic links and junctions")
        if path == root: break
    # Check descendants along the requested path as well.
    path = root
    for part in Path(relative).parts:
        path /= part
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)(): raise ValueError("Linked target refused")
    return target


def directory_size(root):
    root = Path(root)
    if not root.exists(): return 0
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files: safe_path(root, (Path(base) / name).relative_to(root))
        total += sum((Path(base) / name).stat().st_size for name in files)
    return total


def check_quota(root, limit, reserve=0):
    if type(limit) is not int or limit < 1: raise ValueError("Storage quota must be a positive integer")
    if directory_size(root) + reserve > limit: raise StorageLimit("Cuota de almacenamiento agotada; ejecuta mantenimiento o amplía la cuota")


@contextmanager
def offline(manager, ingestion_root):
    # Same lock order as publication's ownership guard: catalog, then job store.
    from ingestion import JobStore
    with manager._lock, manager.transaction() as catalog_db:
        store = JobStore(ingestion_root)
        with store.db() as jobs:
            jobs.execute("BEGIN IMMEDIATE")
            store.expire(jobs)
            if jobs.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone():
                raise RuntimeError("Hay trabajos activos; espera a que terminen antes del mantenimiento")
            yield catalog_db, jobs


def cleanup(manager, ingestion_root, *, apply=False, keep_versions=3, days=30):
    if type(keep_versions) is not int or keep_versions < 1 or days < 1: raise ValueError("Unsafe retention policy")
    ingestion_root = Path(ingestion_root)
    cutoff = time.time() - days*86400
    with offline(manager, ingestion_root) as (db, jobs):
        current = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()
        history = db.execute("SELECT version,created FROM history ORDER BY created DESC").fetchall()
        keep = {row[0] for row in history[:keep_versions]} | ({current[0]} if current else set())
        remove_versions = {version for version, created in history if version not in keep and created < cutoff}
        retained = {version for version, _ in history} - remove_versions
        referenced = set()
        for version in retained:
            snapshot = manager._load(version)  # Any unreadable retained snapshot aborts deletion.
            referenced.update(row.get("audio_file") for row in snapshot._rows if row.get("audio_file"))
        candidates = []
        for area in ("versions", "staging"):
            parent = manager.root / area
            if parent.exists():
                for item in parent.iterdir():
                    path = safe_path(manager.root, item.relative_to(manager.root))
                    if not HEX.fullmatch(item.name): continue
                    if item.stat().st_mtime < cutoff and (area == "staging" or item.name not in retained):
                        candidates.append((manager.root, path, "generation"))
        for area in ("work", "media"):
            parent = ingestion_root / area
            if parent.exists():
                for item in parent.iterdir():
                    path = safe_path(ingestion_root, item.relative_to(ingestion_root))
                    identity = item.stem if area == "media" else item.name
                    if not HEX.fullmatch(identity) or item.stat().st_mtime >= cutoff: continue
                    if area == "media" and (item.suffix != ".mp3" or item.name in referenced): continue
                    candidates.append((ingestion_root, path, "unreferenced_"+area))
        old_jobs = [row[0] for row in jobs.execute("SELECT id FROM jobs WHERE status IN ('completed','failed','interrupted') AND updated < ?", (cutoff,))]
        report = dict(applied=apply, keep_versions=sorted(retained), expires_audio_versions=sorted(remove_versions),
                      files=[dict(path=str(path), reason=reason) for _, path, reason in candidates], jobs=old_jobs)
        if apply:
            for version in remove_versions: db.execute("DELETE FROM history WHERE version=?", (version,))
            jobs.executemany("DELETE FROM jobs WHERE id=?", [(job,) for job in old_jobs])
    if apply:
        # Expire history before moving files. A crash leaves extra files, not a
        # committed live pointer to files that were removed by maintenance.
        moved = []
        for root, path, _ in candidates:
            quarantine = root / "trash" / uuid4().hex
            quarantine.parent.mkdir(exist_ok=True)
            safe_path(root, path.relative_to(root))
            if path.is_dir(): directory_size(path)
            path.rename(quarantine)
            os.utime(quarantine, None)  # The recovery interval starts at quarantine time.
            moved.append(dict(original=str(path), quarantine=str(quarantine)))
        report["quarantine"] = moved
        if moved:
            record = manager.root / "trash" / (uuid4().hex + ".json")
            record.write_text(json.dumps(moved, indent=2), encoding="utf-8")
    # Quarantine is intentionally retained for manual recovery; purge is a separate explicit operation.
    return report


def backup(manager, ingestion_root, destination):
    from catalog import FILES, digest
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists(): raise FileExistsError(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, created=time.time(), sha256={})
    try:
        with offline(manager, ingestion_root) as (db, jobs), zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
            versions = [row[0] for row in db.execute("SELECT version FROM history")]
            payloads = {"catalog/registry.sqlite3": db.serialize(), "ingestion/jobs.sqlite3": jobs.serialize()}
            sources = {}
            for version in versions:
                snapshot = manager._load(version)
                for name in (*FILES, "manifest.json"):
                    path = safe_path(manager.root, Path("versions") / version / name)
                    sources[f"catalog/versions/{version}/{name}"] = path
                for row in snapshot._rows:
                    if row.get("audio_file"):
                        path = safe_path(ingestion_root, Path("media") / row["audio_file"])
                        sources[f"ingestion/media/{path.name}"] = path
            limit = int(os.getenv("SPELLFIND_BACKUP_QUOTA_BYTES", str(2*1024**3)))
            if sum(map(len, payloads.values())) + sum(path.stat().st_size for path in sources.values()) > limit:
                raise StorageLimit("La copia supera la cuota configurada")
            for name, data in payloads.items():
                archive.writestr(name, data)
                manifest["sha256"][name] = hashlib.sha256(data).hexdigest()
            for name, source in sources.items():
                archive.write(source,name)
                manifest['sha256'][name] = digest(source)
            archive.writestr("backup.json", json.dumps(manifest))
        verify_backup(temporary)
        temporary.rename(destination)
    except Exception:
        if temporary.exists(): temporary.unlink()
        raise
    return dict(path=str(destination), files=len(manifest["sha256"]), sha256=digest(destination))


def verify_backup(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("backup.json"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sha256"), dict): raise ValueError("Invalid backup manifest")
        if len(names) != len(set(names)) or set(names) != set(manifest["sha256"]) | {"backup.json"}: raise ValueError("Invalid backup inventory")
        limit = int(os.getenv("SPELLFIND_BACKUP_QUOTA_BYTES", str(2*1024**3)))
        if sum(info.file_size for info in archive.infolist()) > limit + 1024**2: raise StorageLimit("Backup expansion limit")
        for name, sha in manifest["sha256"].items():
            if "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts or not name.startswith(("catalog/", "ingestion/")):
                raise ValueError("Unsafe archive path")
            checksum = hashlib.sha256()
            with archive.open(name) as source:
                for chunk in iter(lambda: source.read(1024*1024), b''): checksum.update(chunk)
            if checksum.hexdigest() != sha: raise ValueError("Corrupt backup")
        return manifest


def restore(path, destination):
    from catalog import CatalogManager
    manifest = verify_backup(path)
    destination = Path(destination)
    if destination.exists(): raise FileExistsError("Restore requires a new destination")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for name in manifest["sha256"]:
                target = safe_path(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source,output,1024*1024)
        for database in ("catalog/registry.sqlite3", "ingestion/jobs.sqlite3"):
            with closing(sqlite3.connect(destination / database)) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise ValueError("Invalid restored database")
        manager = CatalogManager(destination / "catalog", destination / "unused-legacy")
        with manager.transaction() as db: versions = [row[0] for row in db.execute("SELECT version FROM history")]
        for version in versions:
            snapshot = manager.at_version(version)
            for row in snapshot._rows:
                if row.get("audio_file") and not safe_path(destination / "ingestion", Path("media") / row["audio_file"]).is_file():
                    raise ValueError("Restored media is missing")
        return dict(path=str(destination), catalog_version=manager.current().version, generations=len(versions))
    except Exception:
        # Preserve failed restores for diagnosis; never switch the live service.
        raise


def main():
    from catalog import default_manager
    from config import INGEST_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["cleanup", "backup", "restore", "purge-trash"])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--keep-versions", type=int, default=3)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    manager = default_manager()
    if args.action == "cleanup": result = cleanup(manager, INGEST_DIR, apply=args.apply, days=args.days, keep_versions=args.keep_versions)
    elif args.action == "backup":
        if args.path is None: parser.error("backup requires --path")
        result = backup(manager, INGEST_DIR, args.path)
    elif args.action == "restore":
        if args.path is None or args.destination is None: parser.error("restore requires --path and --destination")
        result = restore(args.path, args.destination)
    else:
        cutoff, removed = time.time() - max(1, args.days)*86400, []
        with offline(manager, INGEST_DIR):
            for root in (manager.root, INGEST_DIR):
                trash = root / "trash"
                if not trash.exists(): continue
                for item in trash.iterdir():
                    identity = item.stem if item.suffix == ".json" else item.name
                    if HEX.fullmatch(identity) and item.stat().st_mtime < cutoff:
                        target = safe_path(root, item.relative_to(root))
                        removed.append(str(target))
                        if args.apply:
                            if target.is_dir():
                                directory_size(target)  # reject links throughout the tree
                                shutil.rmtree(target)
                            else: target.unlink()
        result = dict(applied=args.apply, files=removed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
