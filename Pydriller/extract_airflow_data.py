from pydriller import Repository
from datetime import datetime
import json

# Define the repository URL
repo_url = "https://github.com/apache/airflow"

# Define the extraction period
start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 1, 1)

def extract_data():
    results = []
    
    # Initialize the Repository miner with the specified period
    # Note: pydriller will clone the repo locally if it's not a local path
    repo_miner = Repository(repo_url, since=start_date, to=end_date)
    
    print(f"Starting extraction from {repo_url}...")
    print(f"Period: {start_date} to {end_date}")

    count = 0
    for commit in repo_miner.traverse_commits():
        commit_data = {
            # Requested Commit Object fields
            "msg": commit.msg,
            "author": {
                "name": commit.author.name,
                "email": commit.author.email
            },
            "co_authors": [
                {"name": ca.name, "email": ca.email} for ca in commit.co_authors
            ],
            "committer": {
                "name": commit.committer.name,
                "email": commit.committer.email
            },
            "author_date": commit.author_date.isoformat(),
            "deletions": commit.deletions,
            "insertions": commit.insertions,
            "lines": commit.lines,
            "files": commit.files,
            "hash": commit.hash, # Added for reference
            "modified_files": []
        }

        # Extract 'everything' from ModifiedFile objects
        for file in commit.modified_files:
            try:
                sc = file.source_code
                sc_before = file.source_code_before
            except ValueError:
                # specific blob missing, skip this file
                continue

            file_data = {
                "filename": file.filename,
                "old_path": file.old_path,
                "new_path": file.new_path,
                "change_type": str(file.change_type),
                "diff": file.diff,
                "diff_parsed": file.diff_parsed,
                "added_lines": file.added_lines,
                "deleted_lines": file.deleted_lines,
                "source_code": sc,
                "source_code_before": sc_before,
                "nloc": file.nloc,
                "complexity": file.complexity,
                "token_count": file.token_count,
                # Serialize methods to be JSON serializable
                "methods": [
                    {
                        "name": m.name,
                        "long_name": m.long_name,
                        "start_line": m.start_line, 
                        "end_line": m.end_line,
                        "complexity": m.complexity,
                        "nloc": m.nloc,
                        "params": m.parameters
                    } for m in file.methods
                ],
                "changed_methods": [
                    {
                        "name": m.name,
                        "long_name": m.long_name,
                         "start_line": m.start_line, 
                        "end_line": m.end_line
                    } for m in file.changed_methods
                ]
            }
            commit_data["modified_files"].append(file_data)
        
        results.append(commit_data)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} commits...")

    # Save to a JSON file (This file might be very large given 'source_code' is included)
    output_file = "airflow_commits_data.json"
    print(f"Finished. Saving {count} commits to {output_file}...")
    
    # Using a custom default handler for datetime objects just in case
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, default=str)
    
    print("Done!")

if __name__ == "__main__":
    extract_data()
