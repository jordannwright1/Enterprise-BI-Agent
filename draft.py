    try:
        ast.parse(code)
        
        # --- DYNAMIC DEPENDENCY & SAVING ---
        
        final_packages = extract_dependencies(code)

        if final_packages:
            ensure_packages(final_packages)

        # COMMIT TO DATABASE
        existing_skills = list_all_skills(db_path)
        task_id = get_skill_name(task, existing_skills)
        save_skill(task_id, task, code, final_packages)
        print(f"💾 Skill Saved Successfully: {task_id}")

        return {
            "generated_tool_code": code,
            "packages": final_packages,
            "last_error": None,
            "plan": state.get('plan', []) + [f"### 🛠 Skill Saved\nTask: {task_id}"]
        }


pkg_str = " ".join(packages)


  



    
db_path = "./tools/navi_skills.db"


packages_to_install = state.get("packages", [])
    pkg_str = " ".join(packages_to_install)
    install_cmd = f"pip install --no-cache-dir {pkg_str} --quiet --root-user-action=ignore &&" if pkg_str else ""
    docker_command = f"{install_cmd} echo {encoded_payload} | base64 -d | python3"
