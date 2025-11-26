# 📑 DOCUMENTATION INDEX

**Tất cả tài liệu của dự án được tổ chức ở đây**

---

## ⭐ START HERE (Root Folder)

### 🚀 Main Documents (Read in this order)

1. **[START_HERE.md](START_HERE.md)** ← BEGIN HERE!
   - Quick navigation guide
   - 3-step getting started
   - Quick commands

2. **[ROADMAP.md](ROADMAP.md)** ⭐ MOST IMPORTANT
   - Detailed phase-by-phase execution plan
   - Checklists for each step
   - Progress tracking
   - Test commands & validation
   - **USE THIS AS YOUR MAIN GUIDE!**

3. **[ARCHITECTURE.md](ARCHITECTURE.md)**
   - System architecture (Vietnamese)
   - 3-layer design explained
   - Outcome model concept
   - Dataset specification

4. **[README.md](README.md)**
   - Project overview
   - Quick start commands
   - Tech stack

5. **[SETUP_GUIDE.md](SETUP_GUIDE.md)**
   - Detailed setup instructions
   - Environment configuration
   - Troubleshooting

6. **[PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md)**
   - High-level master plan
   - Phase overview
   - Prompt template for handoff

---

## 📚 Additional Docs (docs/ Folder)

### Reference Documents

7. **[docs/notes.md](docs/notes.md)**
   - Development notes
   - Known issues & solutions
   - Performance benchmarks
   - TODO backlog

8. **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**
   - Detailed folder structure explanation
   - File count summary
   - Navigation guide

### Archive (Old planning docs - for reference only)

- [docs/NEW_STRUCTURE_FINAL.md](docs/NEW_STRUCTURE_FINAL.md)
- [docs/README_NEW_STRUCTURE.md](docs/README_NEW_STRUCTURE.md)
- [docs/RESTRUCTURE_PLAN.md](docs/RESTRUCTURE_PLAN.md)
- [docs/SIMPLE_STRUCTURE.md](docs/SIMPLE_STRUCTURE.md)

*(These were planning documents, now archived)*

---

## 🔍 Which Document for What?

| I want to... | Read this |
|--------------|-----------|
| 🚀 Get started quickly | [START_HERE.md](START_HERE.md) |
| 📋 Follow step-by-step guide | [ROADMAP.md](ROADMAP.md) ⭐ |
| 🏗️ Understand architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| ⚙️ Set up environment | [SETUP_GUIDE.md](SETUP_GUIDE.md) |
| 📖 Get project overview | [README.md](README.md) |
| 🔍 Find specific files | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |
| 🐛 Troubleshoot issues | [docs/notes.md](docs/notes.md) |
| 📝 See master plan | [PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md) |

---

## 📂 Code Documentation

### Layer-specific docs

- **Layer 1 (NinjaTrader)**
  - [src/layer1_ninjatrader/README.md](src/layer1_ninjatrader/README.md)
  - Setup guide for NinjaTrader strategy

- **Layer 2 (Feature Engine)**
  - Code comments in each module
  - API server: `src/layer2_feature_engine/api_server.py`

- **Layer 3 (Model)**
  - Code comments in training/inference scripts
  - Model architectures in `src/layer3_model/training/train_outcome_model.py`

---

## 🎯 Recommended Reading Order

### For New Users:
1. START_HERE.md (5 min)
2. README.md (10 min)
3. ARCHITECTURE.md (20 min)
4. ROADMAP.md (start working through phases)

### For Experienced Users:
- Jump straight to ROADMAP.md and begin Phase 1

### For Maintenance:
- docs/notes.md for known issues
- docs/PROJECT_STRUCTURE.md for file locations

---

## 📊 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| START_HERE.md | ✅ Active | 2025-01-26 |
| ROADMAP.md | ✅ Active | 2025-01-26 |
| ARCHITECTURE.md | ✅ Active | 2025-01-26 |
| README.md | ✅ Active | 2025-01-26 |
| SETUP_GUIDE.md | ✅ Active | 2025-01-26 |
| PROJECT_MASTER_PLAN.md | ✅ Active | 2025-01-26 |
| docs/notes.md | ✅ Active | 2025-01-26 |
| docs/PROJECT_STRUCTURE.md | ✅ Active | 2025-01-26 |
| docs/archive/* | 📦 Archived | - |

---

## 🗂️ Full File Tree

```
Documentation/
│
├── Root (Main docs - READ THESE)
│   ├── START_HERE.md          ⭐ Navigation
│   ├── ROADMAP.md              ⭐⭐⭐ Main guide
│   ├── ARCHITECTURE.md         System design
│   ├── README.md               Overview
│   ├── SETUP_GUIDE.md          Setup instructions
│   ├── PROJECT_MASTER_PLAN.md  Master plan
│   └── INDEX.md                This file
│
├── docs/ (Additional docs)
│   ├── notes.md                Development notes
│   ├── PROJECT_STRUCTURE.md    Structure guide
│   └── [archive]/              Old planning docs
│
└── src/*/README.md (Code-specific)
    └── layer1_ninjatrader/README.md
```

---

## 💡 Tips for Navigation

1. **Bookmark** [START_HERE.md](START_HERE.md) and [ROADMAP.md](ROADMAP.md)
2. **Keep** ROADMAP.md open while working
3. **Update** progress in ROADMAP.md after each phase
4. **Check** docs/notes.md for troubleshooting
5. **Reference** ARCHITECTURE.md when confused about design

---

**Last Updated**: 2025-01-26
**Maintained by**: Project Team
