"""d3-zen-stn-pfc-decision (Zenodo 10.5281/zenodo.4960485 = Dryad mirror; Zavala, Jang & Zaghloul, eLife 2018, doi:10.7554/eLife.31007)
-> BIDS-iEEG DERIVATIVE.
Source: SeqMemRawdata.zip. Per subject: ecogLFPdata_rereferenced/<base>.<chan> = one int16 file per channel (README: 'can be
read as int16'); article: macro (STN, 1.5024 kHz) and subdural (PFC strips) signals band-passed 1-500 Hz, 60 Hz notch,
downsampled to 1 kHz, macro re-referenced to the common average, iEEG bipolar -> derivative. Channel extensions: NNN (single
channel) and NNN-MMM (pair); 0NN = macro/LFP channels, 5NN = subdural channels (by numbering; the release does not map them
to contacts). 'bad' suffix -> status bad. behavioral/session_k/events.mat: events with eegfile + eegoffset (sample) -> events.tsv.
Sampling rate per recording from the events (eegoffset vs mstime). Privacy: recording dates in file names and in
recordingdetails.txt / events (mstime = absolute ms clock, eegfile paths) -> files renamed, dates day 01, mstime made relative.
Usage: python b3w10_convert_pfc.py <zenodo dir> <bids root>"""
import glob, json, os, re, shutil, sys
import numpy as np, scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w10_common import write_run, wtsv, wjson, flag_strings, mat_strings

SRC, OUT = sys.argv[1], sys.argv[2]
X_ = os.path.join(os.path.dirname(os.path.dirname(SRC.rstrip("/"))), "extract", "SeqMemRawdata")
SD = os.path.join(OUT, "sourcedata", "zenodo-4960485", "SeqMemRawdata")
report = {"runs": [], "renamed": {}, "notes": []}
DATE6 = re.compile(r"(DBS\d{3})_(\d\d)(\d\d)(\d\d)_")


def newbase(b):
    return DATE6.sub(lambda q: f"{q[1]}_{q[2]}{q[3]}01_", b)


def scrub_text(t):
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda q: f"{int(q[1]):02d}/01/{q[3]}", t)  # mm/dd/yyyy -> mm/01/yyyy
    t = DATE6.sub(lambda q: f"{q[1]}_{q[2]}{q[3]}01_", t)
    return t


def ev_load(p):
    e = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["events"]
    return list(np.atleast_1d(e))


scans = {}
for sdir in sorted(glob.glob(os.path.join(X_, "DBS*"))):
    sid = os.path.basename(sdir); sub = f"sub-{sid}"
    evs = {}
    for ep in sorted(glob.glob(os.path.join(sdir, "behavioral", "session_*", "events.mat"))):
        sess = int(re.search(r"session_(\d+)", ep).group(1))
        evs[sess] = ev_load(ep)
    files = sorted(glob.glob(os.path.join(sdir, "ecogLFPdata_rereferenced", "*")))
    bases = {}
    for f in files:
        b, ext = os.path.basename(f).rsplit(".", 1)
        bases.setdefault(b, []).append((ext, f))
    for b, chs in sorted(bases.items()):
        # which session's events point to this base
        sess_hits = [(s, [e for e in E if str(getattr(e, "eegfile", "")).endswith(b)]) for s, E in evs.items()]
        sess_hits = [(s, h) for s, h in sess_hits if h]
        sess = sess_hits[0][0] if sess_hits else None
        E = sess_hits[0][1] if sess_hits else []
        fs = None
        if len(E) > 2:
            off = np.array([float(e.eegoffset) for e in E]); ms = np.array([float(e.mstime) for e in E])
            ok = np.isfinite(off) & np.isfinite(ms) & (off > 0)
            if ok.sum() > 2:
                fs_est = np.polyfit(ms[ok] / 1000.0, off[ok], 1)[0]
                fs = 1000.0
                report["notes"].append(f"{b}: sampling rate estimated from events (eegoffset vs mstime) {fs_est:.3f} Hz; written as 1000 Hz (article)")
        if fs is None:
            fs = 1000.0; report["notes"].append(f"{b}: no usable events; 1000 Hz per article")
        def key(x):
            m = re.match(r"(\d+)(?:-(\d+))?(bad)?$", x[0]); return (int(m[1]), int(m[2] or 0))
        chs = sorted(chs, key=key)
        lens = {ext: os.path.getsize(f) // 2 for ext, f in chs}
        n = min(lens.values())
        if len(set(lens.values())) > 1:
            report["notes"].append(f"{b}: channel lengths differ {sorted(set(lens.values()))}; truncated to {n}")
        X = np.stack([np.fromfile(f, dtype="<i2", count=n) for ext, f in chs], axis=1)
        names, types, extra = [], [], []
        for ext, f in chs:
            m = re.match(r"(\d+)(?:-(\d+))?(bad)?$", ext)
            a = int(m[1]); macro = a < 500
            nm = (f"LFP{m[1]}" if macro else f"ECOG{m[1]}") + (f"-{m[2]}" if m[2] else "")
            names.append(nm); types.append("DBS" if macro else "ECOG")
            extra.append({"description": f"release file extension .{ext}; " + ("macroelectrode LFP channel (0NN numbering)" if macro else "subdural PFC strip channel (5NN numbering)")
                          + ("; pair NNN-MMM = difference of the two channels" if m[2] else ""),
                          "status": "bad" if m[3] else "good", "status_description": "marked 'bad' in the release file name" if m[3] else "n/a",
                          "reference": "bipolar" if m[2] else ("common average of macro contacts (article)" if macro else "n/a"),
                          "group": "STN_macro" if macro else "PFC_strip"})
        ses = f"ses-{sess}" if sess is not None else "ses-nobeh"
        nb = newbase(b)
        rec = re.search(r"_\d{6}_(\d+)_", b)
        stem = f"{sub}_{ses}_task-seqmem_run-{int(rec.group(1)) if rec else 1}"
        D = os.path.join(OUT, sub, ses, "ieeg")
        r = write_run(D, stem, X, fs, names, ["n/a"] * len(names), types, {
            "TaskName": "seqmem",
            "TaskDescription": "Sequential working-memory task: eight single-digit numbers presented in sequence; remember only numbers inside the target shape and ignore distractors; verbal report at the end of each block (article).",
            "PowerLineFrequency": 60,
            "SoftwareFilters": {"Bandpass": {"Description": "1-500 Hz (article)"}, "Notch": {"Description": "60 Hz (article)"},
                                "Downsampling": {"Description": "to 1 kHz from 1.5024 kHz (macro) (article)"},
                                "Rereferencing": {"Description": "macro: common average of simultaneously recorded macro contacts; subdural: bipolar (article)"}},
            "HardwareFilters": "n/a",
            "iEEGReference": "Re-referenced (see SoftwareFilters); pairs NNN-MMM are differences",
            "iEEGPlacementScheme": "Intraoperative: three macro/microelectrode pairs in the STN trajectory and two subdural strips (6 and 8 contacts) over anterior and lateral PFC (article); per-subject notes in sourcedata docs/recordingdetails.txt",
            "Manufacturer": "Alpha Omega", "ManufacturersModelName": "MicroGuide Pro (article)",
            "ElectricalStimulation": False,
        }, ch_extra=extra, comment=f"b3w10_convert_pfc.py: Zenodo 4960485 SeqMemRawdata/{sid}/ecogLFPdata_rereferenced/{nb}.*, int16 values unchanged")
        rows = []
        if E:
            t0 = float(E[0].mstime)
            for e in E:
                o = float(e.eegoffset)
                rows.append([round(o / fs, 6), 0, "number", int(e.number) if np.isscalar(e.number) and np.isfinite(e.number) else "n/a",
                             int(o), e.block, e.target, e.circle, e.correct, str(e.saidseq) if np.size(e.saidseq) else "n/a", round(float(e.mstime) - t0, 3)])
        wtsv(os.path.join(D, stem + "_events.tsv"), ["onset", "duration", "trial_type", "value", "sample", "block", "target", "circle", "correct", "saidseq", "mstime_rel"], rows)
        r.update(stem=stem, base_renamed=nb, session=sess, n_events=len(rows), channels=names)
        report["runs"].append(r); scans.setdefault((sub, ses), []).append([f"ieeg/{stem}_ieeg.vhdr", "n/a"])
        print(stem, json.dumps({k: r[k] for k in ("n_samples", "sfreq", "duration_s", "n_channels", "roundtrip_ok", "n_events")}), flush=True)
        for ext, f in chs:
            dst = os.path.join(SD, sid, "ecogLFPdata_rereferenced", f"{nb}.{ext}"); os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(f, dst)
        if nb != b: report["renamed"][b] = nb
    # sourcedata: behaviour (mstime relative, eegfile scrubbed), docs (dates day 01), spikes (renamed)
    for ep in sorted(glob.glob(os.path.join(sdir, "behavioral", "session_*", "events.mat"))):
        raw = sio.loadmat(ep, squeeze_me=False, struct_as_record=True)
        ev = raw["events"]; t0 = float(np.asarray(ev["mstime"][0, 0]).ravel()[0])
        for idx in np.ndindex(ev.shape):
            ev["mstime"][idx] = np.array([[float(np.asarray(ev["mstime"][idx]).ravel()[0]) - t0]]) if np.asarray(ev["mstime"][idx]).size else ev["mstime"][idx]
            s = np.asarray(ev["eegfile"][idx]).ravel()
            if s.size: ev["eegfile"][idx] = np.array([newbase(os.path.basename(str(s[0])))])
        dst = os.path.join(SD, os.path.relpath(ep, X_)); os.makedirs(os.path.dirname(dst), exist_ok=True)
        sio.savemat(dst, {"events": ev}, long_field_names=True)
    for tp in glob.glob(os.path.join(sdir, "docs", "*.txt")):
        dst = os.path.join(SD, os.path.relpath(tp, X_)); os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="latin-1").write(scrub_text(open(tp, encoding="latin-1").read()))
    for tp in glob.glob(os.path.join(sdir, "spikingdata", "**", "*_plexon.txt"), recursive=True):
        rel = newbase(os.path.relpath(tp, X_))
        dst = os.path.join(SD, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(tp, dst)
for (sub, ses), rows in scans.items():
    wtsv(os.path.join(OUT, sub, ses, f"{sub}_{ses}_scans.tsv"), ["filename", "acq_time"], sorted(rows))
for fn in os.listdir(SRC):
    if fn.endswith(".rtf"):
        shutil.copy(os.path.join(SRC, fn), os.path.join(OUT, "sourcedata", "zenodo-4960485", fn))
allstr = set()
for p in glob.glob(os.path.join(SD, "**", "*.mat"), recursive=True): allstr |= mat_strings(p)
for p in glob.glob(os.path.join(SD, "**", "*.txt"), recursive=True):
    if "plexon" not in p: allstr |= set(open(p, encoding="latin-1").read().splitlines())
report["privacy_flags_after"] = flag_strings(allstr)
os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
for fn in (__file__, "b3w10_common.py", "b3w3_common.py"):
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(fn)), os.path.join(OUT, "code", os.path.basename(fn)))
json.dump(report, open(os.path.join(OUT, "code", "conversion_report.json"), "w"), indent=1, default=str)
R = report["runs"]
print(json.dumps({"runs": len(R), "subjects": len({r['stem'].split('_')[0] for r in R}), "hours": sum(r["duration_s"] for r in R) / 3600, "rt_all": all(r["roundtrip_ok"] for r in R),
                  "notes": report["notes"][:60], "privacy_flags_after": report["privacy_flags_after"][:40]}, indent=1))
print("CONVERT_DONE")
