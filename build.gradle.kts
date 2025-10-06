plugins {
    `java-library`
    jacoco
    `maven-publish`
    signing
}

group="de.hatoka.neuralnetwork"
description="Simple Neural Network"
// will be defined by build
// version=

val sonatypeUsername: String? by project
val sonatypePassword: String? by project
val sonatypeRepo: Any by project

repositories {
    gradlePluginPortal()
    mavenCentral()
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
        vendor.set(JvmVendorSpec.ADOPTIUM)
    }

    withJavadocJar()
    withSourcesJar()
}

jacoco {
    toolVersion = "0.8.13"
}

tasks {
    withType<Test> {
        useJUnitPlatform()
        testLogging {
            events("passed", "skipped", "failed")
        }
    }

    withType<JacocoReport> {
        reports {
            xml.required.set(true)
            html.required.set(true)
            html.outputLocation.set( File(project.layout.buildDirectory.asFile.get(), "jacocoHtml") )
        }

        val jacocoTestReport by tasks
        jacocoTestReport.dependsOn("test")
    }
}

publishing {
    publications {
        create<MavenPublication>("hatokaMvn") {
            from(components["java"])

            pom {
                name.set(project.name)
                description.set(project.description)
                url.set("https:/github.com/Thomas-Bergmann/${project.name}")

                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                        distribution.set("repo")
                    }
                }
                developers {
                    developer {
                        id.set("tm")
                        name.set("Thomas Bergmann")
                        email.set("tm@hatoka.de")
                    }
                }

                scm {
                    connection.set("https://github.com/Thomas-Bergmann/${project.name}.git")
                    developerConnection.set("git@github.com:Thomas-Bergmann/${project.name}.git")
                    url.set("https://github.com/Thomas-Bergmann/${project.name}")
                }
            }
        }
    }
    repositories {
        maven {
            url = uri(sonatypeRepo)
            credentials {
                username = sonatypeUsername
                password = sonatypePassword
            }
        }
    }
}

signing {
    sign(publishing.publications["hatokaMvn"])
}

val junitVersion = "5.13.3"
dependencies {
    implementation("org.ejml:ejml-all:0.44.0")
    implementation("com.google.code.gson:gson:2.13.1")
    implementation("org.slf4j:slf4j-api:2.0.17")
    runtimeOnly("ch.qos.logback:logback-classic:1.5.19")

    testImplementation("org.junit.jupiter:junit-jupiter-api:${junitVersion}")
    testImplementation("org.junit.jupiter:junit-jupiter-params:${junitVersion}")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:${junitVersion}")
}
