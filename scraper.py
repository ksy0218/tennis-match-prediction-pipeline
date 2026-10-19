import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
uc.Chrome.__del__ = lambda self: None


def get_player_stats(player_name):
    options = uc.ChromeOptions()
    options.add_argument('--window-size=1920,1080')

    print("Launching stealth browser...")
    # added version_main=154 and added use_subprocess=True to fix winerror 6
    driver = uc.Chrome(options=options, version_main=154, use_subprocess=True)

    try:
        print(f"\nNavigating to ATP Tour to search for '{player_name}'...")
        driver.get('https://www.atptour.com/en/players?matchType=Singles&rank=Top%2010&region=all')

        wait = WebDriverWait(driver, 30)

        # finding the right search box and typing the player name in
        search_box = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Search all ATP players']")))
        search_box.clear()
        search_box.send_keys(player_name)

        time.sleep(1.5)

        first_result = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.player-list ul li")))

        print("Clicking search result...")
        driver.execute_script("arguments[0].click();", first_result)

        # makes sure we are in the individual player page and not the player search page
        wait.until(lambda d: "players/" in d.current_url and "players?" not in d.current_url)
        current_url = driver.current_url.replace('/overview', '/rankings-breakdown?matchType=sgl&rankType=roll')
        driver.get(current_url)
        time.sleep(5)

        points_element = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.atp_player-rank-breakdown div.player-stats-details div.wins"))
        )
        raw_points_text = points_element.text.split('\n')[0]
        player_points = int(raw_points_text.strip())

        rank_element = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.atp_player-rank-breakdown div.player-stats-details div.stat"))
        )
        raw_rank_text = rank_element.text.split('\n')[0]
        player_rank = int(raw_rank_text.strip())

        print(player_points, player_rank)
        return player_points, player_rank

    except Exception as e:
        print(f"Failed during execution: {e}")
        return None


    finally: #bypassing win6 handling error
        try:
            driver.quit()
        except OSError:
            pass


