# Skills Directory

This directory contains the skills system for xagent.

## Structure

```
skills/
├── builtin/              # Built-in skills (committed to git)
│   ├── code_reviewer/
│   │   ├── SKILL.md
│   │   └── template.md
│   └── test_generator/
│       ├── SKILL.md
│       └── template.md
└── manager.py           # Skill manager implementation
```

## Built-in Skills

Built-in skills are located in `src/xagent/skills/builtin/` and are committed to the repository.

> **Note:** Currently no built-in skills are included. Add your own skills here to ship with the application.

## User Skills

Users can add custom skills in `.xagent/skills/` (outside the `src/` directory).

User skills:
- Are not committed to git (see `.gitignore`)
- Override built-in skills with the same name
- Can be added without modifying the source code

## Skill Format

Each skill is a directory containing:

- **SKILL.md** (required): Entry point with description, when to use, execution flow
- **template.md** (optional): Prompt template for the skill
- **examples/** (optional): Example files
- **resources/** (optional): Additional resources

See individual skill directories for examples.

## Adding a New Built-in Skill

1. Create a new directory in `src/xagent/skills/builtin/your_skill/`
2. Add `SKILL.md` with the skill description
3. Optionally add `template.md` or other files
4. The skill will be automatically loaded on startup

## Adding a User Skill

1. Create a directory in `.xagent/skills/your_skill/`
2. Add `SKILL.md` and any other files
3. Restart the server or call `POST /api/skills/reload`
