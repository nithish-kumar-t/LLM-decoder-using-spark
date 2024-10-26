val scala3Version = "3.5.0"

lazy val root = project
  .in(file("."))
  .settings(
    name := "LLM-Decoder-",
    version := "1.0.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )
