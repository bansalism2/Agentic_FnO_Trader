import os
import time
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from pyotp import TOTP

# Load environment variables from absolute path
load_dotenv(dotenv_path='./.env')

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
    driver.get(kite.login_url())
    driver.implicitly_wait(10)
    # Enter user ID
    driver.find_element(By.ID, "userid").send_keys(user_id)
    # Enter password
    driver.find_element(By.ID, "password").send_keys(password)
    # Click login
    driver.find_element(By.XPATH, '//button[@type="submit"]').click()
    time.sleep(2)
    # Enter TOTP
    totp = TOTP(totp_secret).now()
    driver.find_element(By.ID, "userid").send_keys(totp)
    # Click continue
    driver.find_element(By.XPATH, '//button[@type="submit"]').click()
    time.sleep(15)  # Increased wait time for redirect
    # Print the current URL for debugging
    print("Current URL after TOTP:", driver.current_url)
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
        with open('access_token.txt', 'w') as f:
            f.write(access_token)
        print("Access token saved to access_token.txt")
    except Exception as e:
        print(f"Error generating access token: {e}")

if __name__ == "__main__":
    main() 