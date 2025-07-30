import os
import time
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pyotp import TOTP

# Load environment variables from absolute path
load_dotenv(dotenv_path='.././.env')

api_key = os.getenv('kite_api_key')
api_secret = os.getenv('kite_api_secret')
user_id = os.getenv('kite_user_id')
password = os.getenv('kite_password')
totp_secret = os.getenv('kite_totp_secret')

if not all([api_key, api_secret, user_id, password, totp_secret]):
    print("Missing one or more required .env variables: kite_api_key, kite_api_secret, kite_user_id, kite_password, kite_totp_secret")
    exit(1)

# Set up KiteConnect
kite = KiteConnect(api_key=api_key)

# Print the login URL for manual inspection
print("Kite login URL:", kite.login_url())

# Set up Selenium Chrome driver (headless optional)
chrome_options = Options()
# chrome_options.add_argument('--headless')  # Uncomment to run headless
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# You may need to adjust the chromedriver path
chromedriver_path = '/opt/homebrew/bin/chromedriver'  # Homebrew default for Apple Silicon
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

def get_request_token():
    print("Opening Kite login page...")
    try:
        driver.get(kite.login_url())
        wait = WebDriverWait(driver, 20) # 20-second explicit wait

        # Enter user ID
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(user_id)
        # Enter password
        wait.until(EC.presence_of_element_located((By.ID, "password"))).send_keys(password)
        # Click login
        wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))).click()
        
        # Enter TOTP
        totp = TOTP(totp_secret).now()
        # Note: Zerodha reuses the 'userid' field for TOTP on the next page
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(totp)
        
        # Click continue - wait for it to be clickable
        wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))).click()
        
        # Wait for the URL to contain "request_token"
        wait.until(EC.url_contains("request_token="))

        # Print the current URL for debugging
        print("Current URL after TOTP:", driver.current_url)
    except Exception as e:
        # --- DEBUGGING: Take a screenshot on failure ---
        failure_screenshot_path = os.path.join(os.path.dirname(__file__), "login_failure.png")
        driver.save_screenshot(failure_screenshot_path)
        print(f"❌ Login process failed. Screenshot saved to: {failure_screenshot_path}")
        raise e # Re-raise the exception so the driver script knows it failed

    # Get request_token from redirected URL
    current_url = driver.current_url
    if 'request_token=' in current_url:
        request_token = current_url.split('request_token=')[1].split('&')[0]
        print(f"Request token obtained: {request_token}")
        return request_token
    else:
        print("Failed to obtain request token. Check credentials and try again.")
        return None

def main():
    request_token = get_request_token()
    driver.quit()
    if not request_token:
        return
    # Exchange request_token for access_token
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        print(f"Access token: {access_token}")
        # Save to file
        with open('../data/access_token.txt', 'w') as f:
            f.write(access_token)
        print("Access token saved to access_token.txt")
    except Exception as e:
        print(f"Error generating access token: {e}")

if __name__ == "__main__":
    main() 