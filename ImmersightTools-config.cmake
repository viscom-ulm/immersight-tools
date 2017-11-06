get_filename_component(ImmersightTools_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} REALPATH)
set(ImmersightTools_INCLUDE_DIRS ${ImmersightTools_ROOT_DIR}/include)
set(ImmersightTools_LIB 
optimized ${ImmersightTools_ROOT_DIR}/lib/Release/ImmersightTools.lib 
debug ${ImmersightTools_ROOT_DIR}/lib/Debug/ImmersightTools.lib
)