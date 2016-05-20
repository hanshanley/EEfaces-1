from django.conf.urls import patterns, url
import views

urlpatterns = [
        url(r'^$', views.index, name='index'),
        url(r'^check-faces/$', views.check_faces, name='check_faces'),
        ]