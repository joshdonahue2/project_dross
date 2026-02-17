import pytest
from playwright.sync_api import Page, expect

@pytest.fixture
def app_url():
    return "http://127.0.0.1:8001"

def test_screenshot_tools(page: Page, app_url):
    page.goto(app_url, timeout=60000)
    page.wait_for_timeout(2000)
    
    # Switch to Tools
    page.click("#nav-tools")
    page.wait_for_selector("#tools-grid .tool-card")
    page.wait_for_timeout(1000)
    
    # Debug dimensions
    dimensions = page.evaluate("""() => {
        const el = document.getElementById('view-tools');
        return {
            scrollHeight: el.scrollHeight,
            clientHeight: el.clientHeight,
            overflowY: window.getComputedStyle(el).overflowY
        };
    }""")
    print(f"\nDEBUG DIMENSIONS: {dimensions}")
    
    # Take screenshot of the whole page
    page.screenshot(path="tools_tab.png", full_page=True)
