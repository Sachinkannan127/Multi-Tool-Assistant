from langchain.tools import tool
from config import settings
import httpx

@tool
def github_tool(action: str, target: str = "") -> str:
    """Interact with GitHub if the user has provided a GitHub Personal Access Token.
    
    Args:
        action (str): The action to perform. Allowed values:
                      "list_repos" - List public repositories for a user.
                      "list_issues" - List recent issues for a repository.
        target (str): The target of the action. 
                      For "list_repos", this should be the GitHub username (e.g. "microsoft").
                      For "list_issues", this should be the repo name (e.g. "microsoft/vscode").
    """
    token = settings.github_token
    if not token:
        return "Error: No GitHub token configured. Please tell the user to add it in the Integrations tab in Settings."
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AgenticAI-Connector/1.0"
    }
    
    try:
        with httpx.Client() as client:
            if action == "list_repos":
                res = client.get(f"https://api.github.com/users/{target}/repos?sort=updated&per_page=10", headers=headers)
                if res.status_code == 200:
                    repos = res.json()
                    if not repos:
                        return f"No repositories found for user {target}."
                    return "Recent Repositories:\n" + "\n".join([f"- {r['full_name']} (Stars: {r['stargazers_count']})" for r in repos])
                return f"GitHub API error ({res.status_code}): {res.text}"
                
            elif action == "list_issues":
                res = client.get(f"https://api.github.com/repos/{target}/issues?state=open&per_page=10", headers=headers)
                if res.status_code == 200:
                    issues = res.json()
                    if not issues:
                        return f"No open issues found for {target}."
                    return f"Open Issues for {target}:\n" + "\n".join([f"- #{i['number']}: {i['title']}" for i in issues])
                return f"GitHub API error ({res.status_code}): {res.text}"
            
            else:
                return f"Unsupported action: {action}. Use 'list_repos' or 'list_issues'."
    except Exception as e:
        return f"Request failed: {str(e)}"
