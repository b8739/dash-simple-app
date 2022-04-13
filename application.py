from app import application, server, cache

from routes import render_page_content

from layout.sidebar.sidebar_callbacks import toggle_collapse, toggle_classname
from layout.navbar.navbar_callbacks import toggle_navbar_collapse


from pages.platform.platform_callbacks import save_first_data

from environment.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK


# if __name__ == "__main__":
#     # cache.clear()

#     application.run_server(
#         host=APP_HOST,
#         # port=APP_PORT,
#         # debug=APP_DEBUG,
#         debug=True,
#         dev_tools_props_check=DEV_TOOLS_PROPS_CHECK,
#     )

if __name__ == "__main__":
    application.run_server(debug=True)
