from app import application, server, cache

from routes import render_page_content

from layout.sidebar.sidebar_callbacks import toggle_collapse, toggle_classname
from layout.navbar.navbar_callbacks import toggle_navbar_collapse

""" Monitoring Callbacks"""

from pages.monitoring.monitoring_callbacks import changeTag

""" Analysis Callbacks"""

from pages.analysis.analysis_callbacks import (
    get_shap_importance,
    draw_shap_bar_graph,
    # draw_shap_dependence_graph,
)

""" Modeling Callbacks"""
from pages.modeling.modeling_callbacks import (
    draw_actual_predict_graph,
)  # 시간 오래 걸려서 잠깐 주석 처리

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
