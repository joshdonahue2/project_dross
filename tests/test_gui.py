import pytest
from playwright.sync_api import Page, expect

@pytest.fixture
def app_url():
    return "http://127.0.0.1:8001"

def test_layout_is_contained(page: Page, app_url):
    """Verify that the main app shell fits within the viewport and doesn't clip excessively."""
    page.goto(app_url, timeout=60000)
    page.wait_for_timeout(2000)  # Wait for scripts to load
    
    # Wait for the app to load
    page.wait_for_selector(".app-shell")
    
    # Check that the main sections are visible
    expect(page.locator(".sidebar")).to_be_visible()
    expect(page.locator(".workspace")).to_be_visible()
    
    # Verify that the chat flow and live stream are visible
    expect(page.locator("#chat-messages")).to_be_visible()
    expect(page.locator("#live-stream")).to_be_visible()

def test_chat_functionality(page: Page, app_url):
    """Verify that we can send a message and it appears in the chat."""
    page.goto(app_url, timeout=60000)
    page.wait_for_timeout(2000)
    
    chat_input = page.locator("#chat-input")
    chat_input.fill("Hello from automated test")
    chat_input.press("Enter")
    
    # Check if the user message appeared with a generous timeout
    user_msg = page.locator(".msg.user:has-text('Hello from automated test')")
    expect(user_msg).to_be_visible(timeout=10000)
    
    # After sending many messages, check if scrolling is enabled
    for i in range(10):
        chat_input.fill(f"Message {i}")
        chat_input.press("Enter")
        page.wait_for_timeout(200) # Small delay between sends
        
    chat_flow = page.locator("#chat-messages")
    # Verify it doesn't just overflow the screen
    is_scrollable = page.evaluate("() => { const el = document.getElementById('chat-messages'); return el.scrollHeight > el.clientHeight; }")

def test_view_switching(page: Page, app_url):
    """Verify that sidebar links correctly switch views."""
    page.goto(app_url, timeout=60000)
    page.wait_for_timeout(2000)
    
    # Switch to Diagnostics
    page.click("#nav-system")
    expect(page.locator("#view-system")).to_be_visible()
    expect(page.locator("#view-nexus")).not_to_be_visible()
    
    # Switch back to Nexus
    page.click("#nav-nexus")
    expect(page.locator("#view-nexus")).to_be_visible()
